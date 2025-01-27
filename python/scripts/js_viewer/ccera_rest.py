#!/usr/bin/env python

import json
import threading
import xmlrpc.client
from tornado.web import Application, RequestHandler
from tornado.ioloop import IOLoop
import datetime
import astropy.coordinates as c
import astropy.units as u

altaz = (0,0)
location = (45.3491, -76.0413, 95 ) #ground elevation
cl = None
timer = None

l = c.EarthLocation(lat=location[0],lon=location[1])

from tornado.web import RequestHandler

def update_pointing():
    try:
        global altaz,timer
        altaz = cl.query_both_axes()
        print("[alt,az] = ",altaz)
        timer = threading.Timer(10.0, update_pointing)
        timer.start()
    except KeyboardInterrupt:
       return

class get_pointing(RequestHandler):
  def get(self):
    self.set_header('Access-Control-Allow-Origin', '*')
    self.set_header('Access-Control-Allow-Methods', 'GET')
    self.set_header('Access-Control-Allow-Headers', 'x-prototype-version,x-requested-with')
    self.set_header('Access-Control-Max-Age', '2520')

    aa = c.SkyCoord(alt=80.79*u.deg,az=0.04*u.deg,obstime=datetime.datetime.now(),frame='altaz',location=l)
    radec = aa.transform_to(frame='fk5')
    ra = radec.ra.value
    dec = radec.dec.value
    galpt = aa.transform_to(frame='galactic')
    gl = galpt.l.value
    gb = galpt.b.value

    self.write({'alt': altaz[0], 'az':altaz[1], 'ra':ra, 'dec':dec, 'gl':gl, 'gb':gb})

class get_position(RequestHandler):
  def get(self):
    self.set_header('Access-Control-Allow-Origin', '*')
    self.set_header('Access-Control-Allow-Methods', 'GET')
    self.set_header('Access-Control-Allow-Headers', 'x-prototype-version,x-requested-with')
    self.set_header('Access-Control-Max-Age', '2520')
    self.write({'lat':location[0],'lon':location[1],'el':location[2]})

if __name__ == '__main__':
    cl = xmlrpc.client.ServerProxy("http://172.22.121.35:9090/")
    update_pointing()
    app = Application([("/pointing", get_pointing),
                       ("/position", get_position)])
    app.listen(3000)
    try:
        IOLoop.instance().start()
    except KeyboardInterrupt:
       timer.cancel()