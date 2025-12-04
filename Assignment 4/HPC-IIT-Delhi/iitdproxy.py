#!/usr/bin/env python3
import sys
import getpass
import signal
import time
from getpass import getpass
from datetime import datetime
import threading
import webbrowser
import ssl
import urllib.request as urllib_request
import urllib.parse as urllib_parse

class Proxy:
    proxy_set = {
        'btech': 22, 'dual': 62, 'diit': 21, 'faculty': 82, 'integrated': 21,
        'mtech': 62, 'phd': 61, 'retfaculty': 82, 'staff': 21, 'irdstaff': 21,
        'mba': 21, 'mdes': 21, 'msc': 21, 'msr': 21, 'pgdip': 21
    }
    google = 'http://www.google.com'

    def __init__(self, username, password, proxy_cat):
        self.username = username
        self.password = password
        self.proxy_cat = proxy_cat
        self.auto_proxy = f"http://www.cc.iitd.ernet.in/cgi-bin/proxy.{proxy_cat}"

        context = ssl._create_unverified_context()
        self.urlopener = urllib_request.build_opener(
            urllib_request.HTTPSHandler(context=context),
            urllib_request.ProxyHandler({'auto_proxy': self.auto_proxy})
        )

        self.proxy_page_address = (
            f"https://proxy{Proxy.proxy_set[proxy_cat]}.iitd.ernet.in/cgi-bin/proxy.cgi"
        )
        self.sessionid = None
        self.loggedout = True
        self.new_session_id()
        self.details()

    def is_connected(self):
        proxies = {
            'http': f"http://proxy{Proxy.proxy_set[self.proxy_cat]}.iitd.ernet.in:3128"
        }
        opener = urllib_request.build_opener(
            urllib_request.ProxyHandler(proxies)
        )
        try:
            with opener.open(Proxy.google) as resp:
                response = resp.read().decode("utf-8", errors="ignore")
        except Exception:
            return "Not Connected"

        if "<title>IIT Delhi Proxy Login</title>" in response:
            return "Login Page"
        elif "<title>Google</title>" in response:
            return "Google"
        else:
            return "Not Connected"

    def get_session_id(self):
        try:
            response = self.open_page(self.proxy_page_address)
        except Exception:
            return None

        check_token = 'sessionid" type="hidden" value="'
        try:
            token_index = response.index(check_token) + len(check_token)
        except ValueError:
            return None

        sessionid = ""
        for i in range(16):
            sessionid += response[token_index + i]
        return sessionid

    def new_session_id(self):
        self.sessionid = self.get_session_id()
        self.loginform = {
            'sessionid': self.sessionid,
            'action': 'Validate',
            'userid': self.username,
            'pass': self.password
        }
        self.logout_form = {
            'sessionid': self.sessionid,
            'action': 'logout',
            'logout': 'Log out'
        }
        self.loggedin_form = {
            'sessionid': self.sessionid,
            'action': 'Refresh'
        }

    def login(self):
        self.new_session_id()
        status, response = self._login_internal()
        return status, response

    def _login_internal(self):
        response = self.submitform(self.loginform)

        if "Either your userid and/or password does'not match." in response:
            return "Incorrect", response
        elif f"You are logged in successfully as {self.username}" in response:
            self.loggedout = False

            def ref():
                if not self.loggedout:
                    res = self.refresh()
                    print("Refresh", datetime.now())
                    if res == 'Session Expired':
                        print("Session Expired Run Script again")
                    else:
                        self.timer = threading.Timer(60.0, ref)
                        self.timer.daemon = True
                        self.timer.start()

            self.timer = threading.Timer(60.0, ref)
            self.timer.daemon = True
            self.timer.start()

            return "Success", response
        elif "already logged in" in response:
            return "Already", response
        elif "Session Expired" in response:
            return "Expired", response
        else:
            return "Not Connected", response

    def logout(self):
        self.loggedout = True
        response = self.submitform(self.logout_form)
        if "you have logged out from the IIT Delhi Proxy Service" in response:
            return "Success", response
        elif "Session Expired" in response:
            return "Expired", response
        else:
            return "Failed", response

    def refresh(self):
        response = self.submitform(self.loggedin_form)
        if "You are logged in successfully" in response:
            if f"You are logged in successfully as {self.username}" in response:
                return "Success", response
            else:
                return "Not Logged In", response
        elif "Session Expired" in response:
            return "Expired", response
        else:
            return "Not Connected", response

    def details(self):
        if VERBOSE:
            for prop, value in vars(self).items():
                print(prop, ": ", value)

    def submitform(self, form):
        data = urllib_parse.urlencode(form).encode("ascii")
        req = urllib_request.Request(self.proxy_page_address, data=data)
        with self.urlopener.open(req) as resp:
            return resp.read().decode("utf-8", errors="ignore")

    def open_page(self, address):
        with self.urlopener.open(address) as resp:
            return resp.read().decode("utf-8", errors="ignore")


STATUS = 0
RESPONSE = 1
VERBOSE = False
user = None  # will be set in main


def signal_handler(sig, frame):
    global user
    if user is not None:
        print('\nLogout', user.logout()[STATUS])
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    n = len(sys.argv)
    if n == 1:
        print("\n\nUsage: python3 login_terminal.py file  # file has: username proxy_category\n\n")
    else:
        with open(sys.argv[1], 'r') as f:
            line = f.readlines()[0].strip().split(' ')
        uname = line[0]
        proxycat = line[1]
        passwd = getpass()

        user = Proxy(username=uname, password=passwd, proxy_cat=proxycat)
        login_status = user.login()[STATUS]
        print('\nLogin', login_status)
        if login_status == "Success":
            # Keep process alive to allow periodic refresh + Ctrl+C logout
            try:
                signal.pause()
            except AttributeError:
                # signal.pause() not available on some platforms (e.g. Windows)
                while True:
                    time.sleep(1)