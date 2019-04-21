#making a webapp to test my model on
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import flask

import numpy as np
import tensorflow as tf
import os
import sys
from flask import render_template, send_from_directory, request, redirect, url_for
from werkzeug import secure_filename
from flask import jsonify

app = flask.Flask(__name__)
UPLOAD_FOLDER = 'uploads'

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(os.path.join(UPLOAD_FOLDER, file_name), input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def cleanup(file_name):
  os.remove(os.path.join(UPLOAD_FOLDER,file_name))

#def load_graph(trained_model):   
#    with tf.gfile.GFile(trained_model, "rb") as f:
#        graph_def = tf.GraphDef()
#        graph_def.ParseFromString(f.read())
# 
#    with tf.Graph().as_default() as graph:
#        tf.import_graph_def(
#            graph_def,
#            input_map=None,
#            return_elements=None,
#            name=""
#            )
#    return graph

#@app.route('/')
#def index():
#    return "Webserver is running"

@app.route('/',methods=['POST','GET'])
def index():
    if request.method == 'POST':
       graph = app.graph
       label_file = "./win-test_labels.txt"
       input_height = 299
       input_width = 299
       input_mean = 0
       input_std = 255
       input_layer = "Placeholder"
       output_layer = "final_result"

       try:
         upload_file = request.files['file']
         file_name = secure_filename(upload_file.filename)
         upload_file.save(os.path.join(UPLOAD_FOLDER, file_name))
       except:
         return jsonify({"Error":"No File Uploaded"})

       input_name = "import/" + input_layer
       output_name = "import/" + output_layer
       input_operation = graph.get_operation_by_name(input_name)
       output_operation = graph.get_operation_by_name(output_name)
       sess = tf.Session(graph=graph)
       try:
         t = read_tensor_from_image_file(
               file_name,
               input_height=input_height,
               input_width=input_width,
               input_mean=input_mean,
               input_std=input_std)
       except:
         cleanup(file_name)
         return jsonify({"Error":"That file is not of the correct type"})

       results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: t})
       results = np.squeeze(results)
       top_k = results.argsort()[-5:][::-1]
       labels = load_labels(label_file)

       conf = str(100 * results[top_k[0]])[0:5]
       res = labels[top_k[0]]
       res = res.upper()
       cleanup(file_name)
       return render_template("result.html", result=res, confidence=conf)

    return render_template("base.html")

app.graph=load_graph('./win-test.pb')
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int("5000"), debug=True, use_reloader=False)
