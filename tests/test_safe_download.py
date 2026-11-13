import _requires_python  # noqa: F401
import ast
import io
import http.server
import os
from pathlib import Path
import socket
import stat
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import safe_download as safe


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.previous = os.getcwd()
        os.chdir(self.temp.name)

    def tearDown(self):
        os.chdir(self.previous)
        self.temp.cleanup()

    def response(self, body=b'bytes', status=200, **headers):
        response = Mock(status=status)
        response.getheader.side_effect = lambda key, default=None: headers.get(key, default)
        response.read1.side_effect = io.BytesIO(body).read1
        connection = Mock()
        connection.getresponse.return_value = response
        return connection, Mock()

    def test_public_address_policy(self):
        for value in ['127.0.0.1', '10.0.0.1', '169.254.169.254', '100.100.100.200',
                      '0.0.0.0', '224.0.0.1', '::1', 'fc00::1', 'fe80::1',
                      '::ffff:8.8.8.8', '2002:0808:0808::1', '64:ff9b::808:808']:
            with self.subTest(value=value):
                self.assertFalse(safe._public_address(value))
        self.assertTrue(safe._public_address('8.8.8.8'))
        self.assertTrue(safe._public_address('2606:4700:4700::1111'))

    def test_url_validation_preserves_signed_query(self):
        self.assertEqual(safe._destination('https://storage.example.test/data.zip?sig=a%2Fb&x=1'),
                         ('https', 'storage.example.test', 443, '/data.zip?sig=a%2Fb&x=1'))
        for url in ['file:///tmp/data', 'https://user:pass@example.test/a',
                    'https://example.test:22/a', 'https://example.test:0/a', 'https://[fe80::1%25eth0]/a',
                    'https://example.test/a\n', 'https://example.test/a#fragment',
                    'https://' + 'x' * 254 + '/a']:
            with self.subTest(url=url), self.assertRaises(ValueError):
                safe._destination(url)

    def test_mixed_dns_answers_fail_closed(self):
        rows = [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('8.8.8.8', 443)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('127.0.0.1', 443))]
        with patch.object(socket, 'getaddrinfo', return_value=rows), self.assertRaises(ValueError):
            safe._resolve('example.test', 443, time.monotonic() + 2)

    def test_resolver_failure_is_not_reported_as_policy_rejection(self):
        with patch.object(socket, 'getaddrinfo', side_effect=socket.gaierror('synthetic')), \
                self.assertRaises(OSError) as raised:
            safe._resolve('example.test', 443, time.monotonic() + 2)
        self.assertNotIsInstance(raised.exception, ValueError)

    def test_connection_pins_numeric_address_and_verifies_tls_hostname(self):
        sock = Mock()
        context = Mock()
        context.wrap_socket.return_value = sock
        rows = [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('8.8.8.8', 443))]
        with patch.object(socket, 'socket', return_value=sock), patch.object(socket, 'getaddrinfo') as resolve, \
                patch.object(safe.ssl, 'create_default_context', return_value=context):
            connection, actual = safe._connection('https', 'example.test', 443, rows, time.monotonic() + 2)
            self.assertIs(actual, sock)
            self.assertEqual(connection.host, 'example.test')
            self.assertEqual(connection.default_port, 443)
            connection.request('GET', '/signed?sig=a%2Fb')
            sent = sock.sendall.call_args.args[0]
            self.assertIn(b'Host: example.test\r\n', sent)
            self.assertNotIn(b'Host: example.test:443', sent)
            sock.connect.assert_called_once_with(('8.8.8.8', 443))
            context.wrap_socket.assert_called_once_with(sock, server_hostname='example.test')
            resolve.assert_not_called()

    def test_redirect_revalidates_and_preserves_signed_path(self):
        first = self.response(status=302, Location='https://cdn.example.test/weights?sig=a%2Fb')
        second = self.response(b'model', **{'Content-Length': '5'})
        with patch.object(safe, '_resolve', return_value=['vetted']) as resolve, \
                patch.object(safe, '_connection', side_effect=[first, second]):
            result = safe.download_file('https://example.test/model', max_bytes=10)
        self.assertEqual(Path(result['file_path']).read_bytes(), b'model')
        self.assertEqual([c.args[0] for c in resolve.call_args_list], ['example.test', 'cdn.example.test'])
        self.assertEqual(second[0].request.call_args.args[:2], ('GET', '/weights?sig=a%2Fb'))
        self.assertNotIn('Authorization', second[0].request.call_args.kwargs['headers'])
        first[0].close.assert_called_once()
        second[0].close.assert_called_once()

    def test_private_redirect_is_never_connected(self):
        connection = self.response(status=302, Location='http://127.0.0.1/metadata')
        with patch.object(safe, '_resolve', side_effect=[['public'], ValueError('not public')]), \
                patch.object(safe, '_connection', return_value=connection) as connect, self.assertRaises(ValueError):
            safe.download_file('http://example.test/model', max_bytes=10)
        self.assertEqual(connect.call_count, 1)
        self.assertEqual(list(Path('job_files').iterdir()), [])

    def test_failed_transfers_leave_no_file(self):
        for response in [self.response(status=404), self.response(status=206),
                         self.response(b'123456', **{'Content-Length': '6'}),
                         self.response(b'123456'), self.response(b'123', **{'Content-Length': '4'}),
                         self.response(b'x', **{'Content-Encoding': 'gzip'}),
                         self.response(status=302, Location='http://example.test/down')]:
            with self.subTest(response=response), patch.object(safe, '_resolve', return_value=['public']), \
                    patch.object(safe, '_connection', return_value=response), self.assertRaises(ValueError):
                safe.download_file('https://example.test/model', max_bytes=5)
            self.assertEqual(list(Path('job_files').iterdir()), [])

    def test_redirect_limit_and_expired_deadline(self):
        with patch.object(safe, '_resolve', return_value=['public']), \
                patch.object(safe, '_connection', side_effect=lambda *a: self.response(status=302, Location='/again')), \
                self.assertRaises(ValueError):
            safe.download_file('https://example.test/model', max_bytes=5)
        with self.assertRaises(TimeoutError):
            safe.download_file('https://example.test/model', max_bytes=5, seconds=0)
        self.assertEqual(list(Path('job_files').iterdir()), [])

    def test_real_http_close_chunked_and_slow_body(self):
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                if self.path == '/chunked':
                    self.send_header('Transfer-Encoding', 'chunked')
                self.end_headers()
                try:
                    if self.path == '/slow':
                        for _ in range(50):
                            self.wfile.write(b'x')
                            self.wfile.flush()
                            time.sleep(0.03)
                    elif self.path == '/chunked':
                        self.wfile.write(b'3\r\nabc\r\n2\r\nde\r\n0\r\n\r\n')
                    else:
                        self.wfile.write(b'abcde')
                except (BrokenPipeError, ConnectionResetError):
                    pass

        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        # Only the test resolver may point at loopback. The production policy is unchanged.
        rows = [(socket.AF_INET, socket.SOCK_STREAM, 6, '', server.server_address)]
        try:
            with patch.object(safe, '_resolve', return_value=rows):
                for route in ['/close', '/chunked']:
                    result = safe.download_file('http://example.test' + route, max_bytes=5)
                    self.assertEqual(Path(result['file_path']).read_bytes(), b'abcde')
                    Path(result['file_path']).unlink()
                started = time.monotonic()
                with self.assertRaises((TimeoutError, OSError, ValueError)):
                    safe.download_file('http://example.test/slow', max_bytes=100, seconds=0.15)
                self.assertLess(time.monotonic() - started, 1.5)
                self.assertEqual(list(Path('job_files').iterdir()), [])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def archive(self, members):
        os.makedirs('job_files', exist_ok=True)
        path = Path('job_files/input.zip')
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in members:
                archive.writestr(name, data)
        return path

    def test_nested_images_and_captions_are_flattened(self):
        path = self.archive([('nested/1.PNG', b'image'), ('nested/1.txt', b'caption'),
                             ('__MACOSX/ignored.png', b'metadata'), ('readme.md', b'ignored')])
        destination = Path(safe.extract_dataset(path))
        self.assertEqual(sorted(p.name for p in destination.iterdir()), ['1.PNG', '1.txt'])
        self.assertEqual((destination / '1.txt').read_bytes(), b'caption')

    def test_traversal_links_and_duplicate_names_rejected(self):
        link = zipfile.ZipInfo('link.png')
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        for members in [[('../escape.png', b'x')], [('/absolute.png', b'x')],
                        [('a\\escape.png', b'x')], [('C:/escape.png', b'x')], [(link, b'/etc/passwd')],
                        [('a/1.png', b'x'), ('b/1.PNG', b'y')]]:
            with self.subTest(members=members):
                path = self.archive(members)
                with self.assertRaises(ValueError):
                    safe.extract_dataset(path)
                self.assertEqual(list(Path('job_files').iterdir()), [path])

    def test_expansion_entry_limits_and_crc_failure_cleanup(self):
        path = self.archive([('1.png', b'x' * 100), ('1.txt', b'caption')])
        for name, limit in [('MAX_MEMBER_BYTES', 50), ('MAX_EXTRACTED_BYTES', 50), ('MAX_ARCHIVE_ENTRIES', 1)]:
            with patch.object(safe, name, limit), self.assertRaises(ValueError):
                safe.extract_dataset(path)
            self.assertEqual(list(Path('job_files').iterdir()), [path])
        with patch.object(zipfile.ZipFile, 'open', side_effect=zipfile.BadZipFile('synthetic')), \
                self.assertRaises(zipfile.BadZipFile):
            safe.extract_dataset(path)
        self.assertEqual(list(Path('job_files').iterdir()), [path])

    def test_zip_directory_is_bounded_before_parsing(self):
        path = self.archive([('1.png', b'x')])
        raw = bytearray(path.read_bytes())
        offset = raw.rfind(b'PK\x05\x06')
        raw[offset + 12:offset + 16] = (5 * 1024**2).to_bytes(4, 'little')
        path.write_bytes(raw)
        with patch.object(zipfile, 'ZipFile') as parser, self.assertRaises(safe.ArchiveLimitExceeded):
            safe.extract_dataset(path)
        parser.assert_not_called()
        self.assertEqual(list(Path('job_files').iterdir()), [path])

    def test_corrupt_zip_crc_is_rejected_and_cleaned(self):
        os.makedirs('job_files', exist_ok=True)
        path = Path('job_files/corrupt.zip')
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED) as archive:
            archive.writestr('1.png', b'synthetic')
        raw = bytearray(path.read_bytes())
        raw[30 + len('1.png')] ^= 1
        path.write_bytes(raw)
        with self.assertRaises(zipfile.BadZipFile):
            safe.extract_dataset(path)
        self.assertEqual(list(Path('job_files').iterdir()), [path])

    def test_dataset_success_preserves_archive_and_extract_paths(self):
        path = self.archive([(f'{i}.png', b'image') for i in range(50)] +
                            [(f'{i}.txt', b'caption') for i in range(50)])
        with patch.object(safe, 'download_file', return_value={'file_path': str(path)}) as download:
            result = safe.download_dataset('https://storage.example.test/archive?sig=synthetic')
        self.assertTrue(Path(result['file_path']).is_file())
        self.assertEqual(len(list(Path(result['extracted_path']).iterdir())), 100)
        download.assert_called_once_with('https://storage.example.test/archive?sig=synthetic',
                                        max_bytes=safe.DATASET_MAX_BYTES)

    def test_dataset_extraction_failure_removes_download(self):
        path = self.archive([('../bad.png', b'x')])
        with patch.object(safe, 'download_file', return_value={'file_path': str(path)}), self.assertRaises(ValueError):
            safe.download_dataset('https://storage.example.test/archive')
        self.assertEqual(list(Path('job_files').iterdir()), [])

    def test_handler_uses_safe_dataset_download(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'src/handler.py').read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        self.assertTrue(any(isinstance(n.func, ast.Name) and n.func.id == 'download_dataset' for n in calls))
        self.assertFalse(any(isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                             and n.func.value.id == 'rp_download' for n in calls))


if __name__ == '__main__':
    unittest.main()
