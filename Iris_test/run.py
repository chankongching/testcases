# -*- coding:utf-8 -*-
import tensorflow as tf
import time

flags = tf.app.flags
# 选择日志资料夹
flags.DEFINE_string('data_dir', "/root/code", 'Code location')
# 选择日志资料夹
flags.DEFINE_string('log_dir', "/root/result", 'Directory to store result and logs')
# Defining flags value
FLAGS = flags.FLAGS

def read_data(file_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)
    defaults = [[0], [0.], [0.], [0.], [0.], ['']]
    Id,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species = tf.decode_csv(value, defaults)

    #因为使用的是鸢尾花数据集，这里需要对y值做转换
    preprocess_op = tf.case({
        tf.equal(Species, tf.constant('Iris-setosa')): lambda: tf.constant(0),
        tf.equal(Species, tf.constant('Iris-versicolor')): lambda: tf.constant(1),
        tf.equal(Species, tf.constant('Iris-virginica')): lambda: tf.constant(2),
    }, lambda: tf.constant(-1), exclusive=True)

    return tf.stack([SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]), preprocess_op

def create_pipeline(filename, batch_size, num_epochs=None):
    file_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
    example, label = read_data(file_queue)

    min_after_dequeue = 1000
    capacity = min_after_dequeue + batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )

    return example_batch, label_batch

def convert_to_float(data_set, mode):
    new_set = []
    try:
        if mode == 'training':
            for data in data_set:
                new_set.append([float(x) for x in data[:len(data)-1]] + [data[len(data)-1]])

        elif mode == 'test':
            for data in data_set:
                new_set.append([float(x) for x in data])

        else:
            print('Invalid mode, program will exit.')
            exit()

        return new_set

    except ValueError as v:
        print(v)
        print('Invalid data set format, program will exit.')
        exit()


x_train_batch, y_train_batch = create_pipeline(FLAGS.data_dir + '/Iris-train.csv', 50, num_epochs=1000)
x_test, y_test = create_pipeline(FLAGS.data_dir + '/Iris-test.csv', 60)

init_op = tf.global_variables_initializer()
local_init_op = tf.local_variables_initializer()  # local variables like epoch_num, batch_size
with tf.Session() as sess:
    sess.run(init_op)
    sess.run(local_init_op)

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Retrieve a single instance:
    try:
        #while not coord.should_stop():
        local_step = 0
        while True:
            # x_train_batch_float = convert_to_float(x_train_batch,'training')
            # y_train_batch_float = convert_to_float(y_train_batch,'training')
            example, label = sess.run([x_train_batch, y_train_batch])
            local_step += 1
            now = time.time()
            # print ("Example = ")
            # print (example)
            # print ("label = ")
            # print (label)

            # For Complete session control
            # print '%f: Worker %d: traing step %d dome (global step:%d)' % (now, FLAGS.task_index, local_step, step)

            # For distributed task is available
            # print '%f: Worker %d: training step %d done' % (now, FLAGS.task_index, local_step)

            # Just to print the count
            print '%f: training step %d done' % (now, local_step)
    except tf.errors.OutOfRangeError:
        print ('Done reading')
    finally:
        coord.request_stop()

    coord.join(threads)
    save_path = saver.save(sess, FLAGS.log_dir + '/model.ckpt')
    print("Model saved in path: %s" % save_path)
    sess.close()
