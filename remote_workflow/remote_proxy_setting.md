# 目标
你为什需要这篇文档？
答：由于远程服务器没有访问外网的权限，但是我们的`ai agent`例如`codex`, `claude code`等等，需要访问外网才能正常工作，所以本文档目标是能让服务器能够通过代理中转的方式去访问外网，从而让这些`ai agent`正常工作。

文档目标：让remote server通过本机 Mac 的飞连 VPN 系统路由 出口访问vpn网络。
## 一键配置
直接将本文档发送给本机的`trae-cn`的`@SOLO Coder`，或者`codex`，`claude code`等等AI agent，让它们帮你搞定!

**特别注意**：如果使用`ai agent`的一键配置，则需要将`codex`, `claude code`开启`yolo`模式。`trae-cn`需要在settings内的conversation的Auto-Run模式全部开启。否则这些`agent`会启用`sandbox`模式，无法使用真实的网络环境！

另外：如果你拥有clash等等自己的代理，也可以选择使用自己的代理，不过飞连vpn不需要付费，所以建议使用飞连vpn。
# 配置流程
## 前置条件
- Mac 已连接飞连 VPN，飞连vpn开启了全局模式，且 Mac 本机直连可访问：
  - `curl --max-time 12 https://www.youtube.com` 成功。
- remote server 以及配置好在`~/.ssh/config`内.
## 1. Mac 启动临时 CONNECT 代理（仅走系统路由）
在一个终端上执行以下命令，这个命令会启动一个简易的 HTTP 代理服务器（HTTP Proxy Server），监听端口为`18080`。用于转发服务器的CONNECT请求。
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
验证本机代理可用（另开一个终端）：
```bash
curl -I --max-time 12 -x http://127.0.0.1:18080 https://www.youtube.com
```

## 2. 配置 remote server 上的反向代理
你的 ssh config 配置了如下反代理规则：
在`~/.ssh/config`内，添加如下内容：最重要的是`RemoteForward 18080 localhost:18080`，其他内容根据实际情况填写。
假设我现在的服务器是`910c_1`，那么我需要添加如下内容，请根据你的实际情况填写：
```
Host 910c_1
    HostName 180.184.39.97
    User root
    IdentityFile /Users/bytedance/.ssh/id_ed25519
    ProxyJump jumpecs-proxy-hl
    RemoteForward 18080 127.0.0.1:18080
```

## 3. 验收规则
在 910c_1 上执行验收命令（另开一个终端）：
```bash
ssh 910c_1 "curl --max-time 12 -x http://127.0.0.1:18080 https://www.youtube.com"
```

## 4. 使用代理
如果返回成功，你就可以在服务器的终端使用`export all_proxy=http://127.0.0.1:18080` 来访问mac飞连的vpn网络。然后你的`ai agent`就可以正常工作了。

## 5. 清理与退出(可选，只有当你不再需要访问飞连的vpn网络时才退出)
用`ctrl+c`退出步骤 1 启动的简易 HTTP 代理服务器。