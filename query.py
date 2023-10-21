import socket


class Client():
    def __init__(self, host="127.0.0.1", port=4001, serv_host="127.0.0.1", serv_port=5000) -> None:
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.host,self.port))
        self.server = (serv_host, serv_port)

    def send(self, msg):
        self.sock.sendto(msg.encode("utf-8"), self.server)

    def close(self):
        self.sock.close()


def main():
    c = Client()

    while True:
        query = input("[*] enter query: ")
        if query == "q":
            break
        c.send(query)
    c.close()

if __name__ == "__main__":
    main()