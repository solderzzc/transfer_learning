import tensorflow as tf
import os, sys

bottleneck_file = 'bottlenecks/Leopard/pic1121_Clouded Leopard 159074_resized.jpg.txt'

def get_bottleneck(bottleneck_file):
  with open(bottleneck_file, 'r') as bottleneck_file:
    bottleneck_string = bottleneck_file.read()
  did_hit_error = False
  try:
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  except:
    print("Invalid float found, recreating bottleneck")
    did_hit_error = True
  if did_hit_error:
    return None
  return bottleneck_values
test_bottleneck =  get_bottleneck(bottleneck_file)
if test_bottleneck is None:
    print('Load bottle file error, exiting')
    os._exit(0)

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("output_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("bottlenecks_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

init_ops = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_ops)
    #ops = sess.graph.get_operations()
    #for m in ops:
    #    print(m.values())
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    predictions = sess.run(softmax_tensor, \
             {'input/BottleneckInputPlaceholder:0': [test_bottleneck]})

    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))
