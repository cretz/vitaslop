//! `vitaslop-desktop serve`: host the embedded browser front end.
//!
//! A static host with the two headers the page needs (cross-origin isolation for
//! shared memory) and nothing else - no games directory, no routes; the page takes
//! its titles by upload like anywhere else. `std::net` and a thread per connection:
//! the whole server is smaller than a dependency's manifest, and it serves one
//! person on one LAN.

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};

mod embedded {
    include!(concat!(env!("OUT_DIR"), "/web_files.rs"));
}

fn mime(path: &str) -> &'static str {
    match path.rsplit_once('.').map(|(_, e)| e).unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "wasm" => "application/wasm",
        "json" => "application/json",
        "webmanifest" => "application/manifest+json",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "ttf" => "font/ttf",
        _ => "application/octet-stream",
    }
}

fn find(path: &str) -> Option<&'static [u8]> {
    embedded::FILES.iter().find(|(p, _)| *p == path).map(|(_, b)| *b)
}

fn handle(mut stream: TcpStream) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut line = String::new();
    if reader.read_line(&mut line).is_err() {
        return;
    }
    // Drain the headers; nothing in them changes the answer.
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h).is_err() || h == "\r\n" || h == "\n" || h.is_empty() {
            break;
        }
    }
    let target = line.split_whitespace().nth(1).unwrap_or("/");
    let path = target.split('?').next().unwrap_or("/").trim_start_matches('/');
    let path = if path.is_empty() { "index.html" } else { path };
    let (status, body, ctype) = match find(path) {
        Some(b) => ("200 OK", b, mime(path)),
        None => ("404 Not Found", &b"not found"[..], "text/plain"),
    };
    let _ = write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\n\
         Cross-Origin-Opener-Policy: same-origin\r\nCross-Origin-Embedder-Policy: require-corp\r\n\
         Cache-Control: no-cache\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(body);
    let _ = stream.flush();
}

pub fn run(host: &str, port: u16) -> Result<(), String> {
    if find("pkg/vitaslop_web.js").is_none() {
        return Err("this binary was built without the web bundle: run `node projects/vitaslop-web/build.mjs` \
                    and rebuild vitaslop-desktop to embed it"
            .into());
    }
    let listener = TcpListener::bind((host, port)).map_err(|e| format!("bind {host}:{port}: {e}"))?;
    println!("serving the vitaslop web front end on http://{host}:{port}/ ({} files embedded)", embedded::FILES.len());
    println!("a phone on this network can open http://<this machine's address>:{port}/ - it must be https or localhost to run titles; see the page's browser checks");
    for stream in listener.incoming().flatten() {
        std::thread::spawn(move || handle(stream));
    }
    Ok(())
}
