from flask import Flask, render_template, send_from_directory, Response
from pathlib import Path
import argparse

from mtg.web.camera import Camera


if __name__ == "__main__":
    cam = Camera()
    cam.run()

    
app_name = __name__
app = Flask(app_name)

@app.after_request
def add_header(index):
    """
    Add headers to both force latest IE rendering or Chrome Frame,
    and also to cache the rendered page for 10 minutes
    """
    index.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    index.headers["Pragma"] = "no-cache"
    index.headers["Expires"] = "0"
    index.headers["Cache-Control"] = "public, max-age=0"
    return index

@app.route("/")
def entrypoint():
    return render_template("index.html")

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')

@app.route("/stream")
def stream():
    global cam
    return Response(gen(cam),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--port', type=int, default=5000, help="Running port")
    parser.add_argument("-H","--host", type=str, default='0.0.0.0', help="Address to broadcast")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port)
