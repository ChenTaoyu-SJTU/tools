# Remote Proxy Setting

## 目标
本文档用于让 remote server 通过本机 Mac 的 VPN 系统路由访问外网，从而让需要联网的 AI agent 正常工作。

## 备注
- 允许 AI agent 一键配置时，必须使用真实网络环境，不能走 sandbox。
- 需要允许 agent 执行真实命令，否则无法完成网络代理配置。
- 如果你已有自己的代理，也可以直接使用；本文档默认使用 Mac 上的飞连 VPN。

## 前提
- Mac 已连接飞连 VPN，且开启全局模式。
- Mac 本机可直接访问外网，例如：
```bash
curl --max-time 12 https://www.youtube.com
```
- remote server 已配置在本地 `~/.ssh/config` 中。

## 1. 在 Mac 上启动临时 CONNECT 代理
在本机终端执行以下命令，监听 `127.0.0.1:18080`：

```bash
PORT=18080
python3 - <<'PY'
import socket, threading, os
HOST='127.0.0.1'
PORT=int(os.environ.get('PORT','18080'))

def relay(a, b):
    try:
        while True:
            d=a.recv(65536)
            if not d: break
            b.sendall(d)
    except: pass
    finally:
        for s in (a,b):
            try: s.shutdown(socket.SHUT_RDWR)
            except: pass
            try: s.close()
            except: pass

def handle(c):
    c.settimeout(10)
    try:
        buf=b''
        while b'\r\n\r\n' not in buf and len(buf)<16384:
            x=c.recv(4096)
            if not x: return
            buf+=x
        head, rest = buf.split(b'\r\n\r\n',1)
        line=head.split(b'\r\n',1)[0].split()
        if len(line)<2 or line[0].upper()!=b'CONNECT':
            c.sendall(b"HTTP/1.1 405 Method Not Allowed\r\nContent-Length:0\r\n\r\n")
            return
        hostport=line[1].decode()
        host, port = (hostport.rsplit(':',1)+['443'])[:2]
        u=socket.create_connection((host,int(port)),timeout=10)
        c.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
        if rest: u.sendall(rest)
        t1=threading.Thread(target=relay,args=(c,u),daemon=True)
        t2=threading.Thread(target=relay,args=(u,c),daemon=True)
        t1.start(); t2.start(); t1.join(); t2.join()
    except:
        try: c.sendall(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length:0\r\n\r\n")
        except: pass
    finally:
        try: c.close()
        except: pass

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
s.bind((HOST,PORT))
s.listen(128)
print(f"CONNECT proxy listening on {HOST}:{PORT}", flush=True)
while True:
    c,_=s.accept()
    threading.Thread(target=handle,args=(c,),daemon=True).start()
PY
```

验证本机代理：

```bash
curl --max-time 12 -x http://127.0.0.1:18080 https://www.youtube.com
```

## 2. 配置 SSH config 中的 `RemoteForward`
在本地 `~/.ssh/config` 中，为目标 remote server 添加或确认以下配置。

关键项只有这一行：

```text
RemoteForward 18080 127.0.0.1:18080
```

示例：

```text
Host 910c_1
    HostName 180.184.39.97
    User root
    IdentityFile /Users/bytedance/.ssh/id_ed25519
    ProxyJump jumpecs-proxy-hl
    RemoteForward 18080 127.0.0.1:18080
```

## 3. 验收
在 remote server 上执行：

```bash
ssh 910c_1 "curl --max-time 12 -x http://127.0.0.1:18080 https://www.youtube.com"
```

返回成功即表示反向代理可用。

## 4. 使用方式
验收通过后，在服务器终端中执行：

```bash
export all_proxy=http://127.0.0.1:18080
```

之后 remote server 上的 AI agent 就可以通过 Mac 的 VPN 网络访问外网。

## 5. 退出方式
不再需要代理时，回到步骤 1 的本机终端，按 `Ctrl+C` 停止临时 CONNECT 代理。
