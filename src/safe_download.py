"""Bounded public-network downloads and private, bounded dataset extraction."""

import http.client
import ipaddress
import os
from pathlib import Path
import queue
import shutil
import socket
import ssl
import stat
import struct
import tempfile
import threading
import time
from urllib.parse import quote, urljoin, urlsplit
import zipfile

MODEL_MAX_BYTES = 16 * 1024**3
DATASET_MAX_BYTES = 1024**3
DOWNLOAD_SECONDS = 1800
MAX_REDIRECTS = 5
MAX_ARCHIVE_ENTRIES = 1000
MAX_EXTRACTED_BYTES = 1024**3
MAX_MEMBER_BYTES = 32 * 1024**2
EXTRACTION_SECONDS = 120
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tif', '.tiff'}
_RESOLVER_SLOTS = threading.BoundedSemaphore(4)


class DownloadDestinationRejected(ValueError):
    pass


class DownloadSizeExceeded(ValueError):
    pass


class DownloadResponseRejected(ValueError):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class UnsafeArchiveMember(ValueError):
    pass


class ArchiveLimitExceeded(ValueError):
    pass


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('download deadline exceeded')
    return remaining


def _public_address(value):
    address = ipaddress.ip_address(value)
    # Reject transition mechanisms as well as private, link-local and reserved IPs.
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped or address.sixtofour or address.teredo:
            return False
        if address in ipaddress.ip_network('64:ff9b::/96') or address in ipaddress.ip_network('64:ff9b:1::/48'):
            return False
    return address.is_global and not address.is_multicast and not address.is_reserved


def _resolve(host, port, deadline):
    # A stalled system resolver must not hold a job forever or spawn unlimited threads.
    timeout = min(10, _remaining(deadline))
    if not _RESOLVER_SLOTS.acquire(blocking=False):
        raise DownloadDestinationRejected('resolver busy')
    result = queue.Queue(maxsize=1)

    def resolve():
        try:
            result.put(socket.getaddrinfo(host, port, type=socket.SOCK_STREAM))
        except Exception:
            result.put(None)
        finally:
            _RESOLVER_SLOTS.release()

    try:
        threading.Thread(target=resolve, daemon=True).start()
    except BaseException:
        _RESOLVER_SLOTS.release()
        raise
    try:
        addresses = result.get(timeout=timeout)
    except queue.Empty:
        raise TimeoutError('name resolution timed out') from None
    if not addresses or any(not _public_address(row[4][0]) for row in addresses):
        raise DownloadDestinationRejected('download destination is not public')
    return addresses[:16]


def _destination(url):
    if not isinstance(url, str) or len(url) > 16384 or any(ord(c) < 33 or ord(c) == 127 for c in url):
        raise DownloadDestinationRejected('invalid download URL')
    parsed = urlsplit(url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise DownloadDestinationRejected('invalid download URL')
    if parsed.fragment or '\\' in parsed.netloc or '%' in parsed.hostname:
        raise DownloadDestinationRejected('invalid download URL')
    host = parsed.hostname
    # Bound IDNA work before conversion, including per-label conversion.
    if len(host) > 253:
        raise DownloadDestinationRejected('invalid hostname')
    host = host.encode('idna').decode('ascii')
    port = parsed.port if parsed.port is not None else (443 if parsed.scheme == 'https' else 80)
    if port not in (80, 443):
        raise DownloadDestinationRejected('unsupported download port')
    target = quote(parsed.path or '/', safe="/%:@!$&'()*+,;=-._~")
    if parsed.query:
        target += '?' + quote(parsed.query, safe="/%?:@!$&'()*+,;=-._~")
    return parsed.scheme, host, port, target


def _connection(scheme, host, port, addresses, deadline):
    last_error = None
    for family, kind, protocol, _, address in addresses:
        sock = socket.socket(family, kind, protocol)
        try:
            sock.settimeout(min(10, _remaining(deadline)))
            sock.connect(address)  # Connect to the validated numeric address, never resolve again.
            if scheme == 'https':
                context = ssl.create_default_context()
                context.set_alpn_protocols(['http/1.1'])
                sock.settimeout(min(10, _remaining(deadline)))
                sock = context.wrap_socket(sock, server_hostname=host)
            # Preserve the scheme's default Host port, including signed storage requests.
            connection_type = http.client.HTTPSConnection if scheme == 'https' else http.client.HTTPConnection
            options = {'context': context} if scheme == 'https' else {}
            connection = connection_type(host, port, timeout=min(60, _remaining(deadline)), **options)
            connection.sock = sock
            return connection, sock
        except Exception as exc:
            sock.close()
            last_error = exc
    raise OSError('download connection failed') from last_error


def _interrupt(sock):
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass


def download_file(url, *, max_bytes, seconds=DOWNLOAD_SECONDS):
    """Return a generated local path, or remove partial bytes on every failure.

    No ambient proxy, netrc, cookies or authorization headers are used. HTTPS
    authenticates the original hostname while connecting to a vetted numeric IP.
    """
    os.makedirs('job_files', exist_ok=True)
    fd, path = tempfile.mkstemp(dir='job_files', suffix='.download')
    deadline = time.monotonic() + seconds
    try:
        with os.fdopen(fd, 'wb') as output:
            for hop in range(MAX_REDIRECTS + 1):
                scheme, host, port, target = _destination(url)
                addresses = _resolve(host, port, deadline)
                connection, sock = _connection(scheme, host, port, addresses, deadline)
                timer = None
                response = None
                try:
                    timer = threading.Timer(_remaining(deadline), _interrupt, args=(sock,))
                    timer.daemon = True
                    timer.start()
                    sock.settimeout(min(60, _remaining(deadline)))
                    connection.request('GET', target, headers={'Accept-Encoding': 'identity', 'User-Agent': 'training-worker'})
                    response = connection.getresponse()
                    if response.status in (301, 302, 303, 307, 308):
                        location = response.getheader('Location')
                        if not location or hop == MAX_REDIRECTS:
                            raise DownloadDestinationRejected('download redirect limit or missing destination')
                        next_url = urljoin(url, location)
                        if scheme == 'https' and urlsplit(next_url).scheme != 'https':
                            raise DownloadDestinationRejected('insecure download redirect')
                        url = next_url
                        continue
                    if response.status != 200:
                        raise DownloadResponseRejected('download HTTP status rejected', response.status)
                    if response.getheader('Content-Encoding', 'identity').lower() != 'identity':
                        raise DownloadResponseRejected('encoded download body rejected')
                    declared = response.getheader('Content-Length')
                    if declared is not None and (not declared.isascii() or not declared.isdecimal() or int(declared) > max_bytes):
                        raise DownloadSizeExceeded('download size rejected')
                    count = 0
                    while True:
                        sock.settimeout(min(60, _remaining(deadline)))
                        chunk = response.read1(min(1024 * 1024, max_bytes - count + 1))
                        if not chunk:
                            break
                        count += len(chunk)
                        if count > max_bytes:
                            raise DownloadSizeExceeded('download size exceeded')
                        output.write(chunk)
                    _remaining(deadline)
                    if declared is not None and count != int(declared):
                        raise DownloadResponseRejected('incomplete download')
                    return {'file_path': os.path.abspath(path)}
                finally:
                    if timer is not None:
                        timer.cancel()
                    if response is not None:
                        response.close()
                    connection.close()
        raise DownloadDestinationRejected('download redirect limit exceeded')
    except BaseException:
        os.unlink(path)
        raise


def _check_zip_directory(path):
    # Bound central-directory allocation before ZipFile constructs its entry list.
    # Dataset limits fit the ordinary ZIP format, so ZIP64/multipart archives are unnecessary.
    with open(path, 'rb') as source:
        source.seek(0, os.SEEK_END)
        length = source.tell()
        source.seek(max(0, length - 65557))
        tail = source.read()
    offset = tail.rfind(b'PK\x05\x06')
    if offset < 0 or len(tail) - offset < 22:
        raise UnsafeArchiveMember('missing ZIP directory')
    _, disk, directory_disk, disk_entries, entries, size, start, comment = struct.unpack_from('<4s4H2LH', tail, offset)
    if (disk or directory_disk or disk_entries != entries or entries == 65535
            or entries > MAX_ARCHIVE_ENTRIES or size > 4 * 1024**2
            or start + size > length or offset + 22 + comment != len(tail)):
        raise ArchiveLimitExceeded('unsupported or excessive ZIP directory')


def extract_dataset(path):
    """Flatten supported raster/caption pairs without trusting archive paths."""
    destination = tempfile.mkdtemp(dir='job_files', prefix='dataset-')
    deadline = time.monotonic() + EXTRACTION_SECONDS
    try:
        _check_zip_directory(path)
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_ARCHIVE_ENTRIES:
                raise ArchiveLimitExceeded('too many archive entries')
            total = 0
            selected = []
            names = set()
            for entry in entries:
                name = entry.orig_filename
                parts = name.rstrip('/').split('/')
                mode = stat.S_IFMT(entry.external_attr >> 16)
                if (not name or '\x00' in name or '\\' in name or ':' in name or name.startswith('/')
                        or any(part in ('', '.', '..') for part in parts)
                        or mode not in (0, stat.S_IFREG, stat.S_IFDIR) or entry.flag_bits & 1):
                    raise UnsafeArchiveMember('unsafe archive member')
                if entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                    raise UnsafeArchiveMember('unsupported archive compression')
                total += entry.file_size
                if entry.file_size > MAX_MEMBER_BYTES or total > MAX_EXTRACTED_BYTES:
                    raise ArchiveLimitExceeded('archive expansion exceeded')
                if entry.is_dir() or '__MACOSX' in parts:
                    continue
                leaf = parts[-1]
                if Path(leaf).suffix.lower() not in IMAGE_SUFFIXES | {'.txt'}:
                    continue
                if leaf.casefold() in names:
                    raise UnsafeArchiveMember('duplicate flattened archive name')
                names.add(leaf.casefold())
                selected.append((entry, leaf))
            for entry, leaf in selected:
                count = 0
                with archive.open(entry) as source, open(os.path.join(destination, leaf), 'xb') as output:
                    while True:
                        _remaining(deadline)
                        chunk = source.read(min(1024 * 1024, MAX_MEMBER_BYTES - count + 1))
                        if not chunk:
                            break
                        count += len(chunk)
                        if count > MAX_MEMBER_BYTES or count > entry.file_size:
                            raise ArchiveLimitExceeded('archive member expansion exceeded')
                        output.write(chunk)
                if count != entry.file_size:
                    raise UnsafeArchiveMember('incomplete archive member')
        return os.path.abspath(destination)
    except BaseException:
        shutil.rmtree(destination)
        raise


def download_dataset(url):
    downloaded = download_file(url, max_bytes=DATASET_MAX_BYTES)
    try:
        downloaded['extracted_path'] = extract_dataset(downloaded['file_path'])
        return downloaded
    except BaseException:
        os.unlink(downloaded['file_path'])
        raise
