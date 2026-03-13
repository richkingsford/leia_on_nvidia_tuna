const http = require('http');
const fs = require('fs');
const path = require('path');
const root = process.cwd();
const server = http.createServer((req, res) => {
  const cleanPath = decodeURIComponent((req.url || '/').split('?')[0]);
  const rel = cleanPath === '/' ? 'forklift_bricks_isometric.svg' : cleanPath.replace(/^\//, '');
  const filePath = path.join(root, rel);
  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end('Not found');
      return;
    }
    const contentType = filePath.endsWith('.svg') ? 'image/svg+xml' : filePath.endsWith('.html') ? 'text/html; charset=utf-8' : filePath.endsWith('.js') ? 'text/javascript; charset=utf-8' : 'text/plain';
    res.writeHead(200, { 'Content-Type': contentType, 'Cache-Control': 'no-store' });
    res.end(data);
  });
});
server.listen(4173, '127.0.0.1', () => console.log('listening'));
setInterval(() => {}, 1 << 30);


