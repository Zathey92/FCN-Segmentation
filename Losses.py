
def sigmoid_cross_entropy(output, target):
    with tf.name_scope("sce_loss"):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output))

def weighted_cross_entropy(output,target,weight):
    with tf.name_scope("wce_loss"):
        return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=a5,targets=self.segm_map,pos_weight = 2))

def binary_cross_entropy(output, target, epsilon=1e-8):

    with tf.name_scope("bce_loss"):
        return tf.reduce_mean(tf.reduce_sum(-(target * tf.log(output + epsilon) + (1. - target) * tf.log(1. - output + epsilon)), axis=1))

def mean_squared_error(output, target, is_mean=False, name="mean_squared_error"):
    #Return the TensorFlow expression of mean squared error (L2) of two batch of data.
    with tf.name_scope("mean_squared_error"):
        if output.get_shape().ndims == 2:  # [batch_size, n_feature]
            if is_mean:
                mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, target), 1))
            else:
                mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output, target), 1))
        elif output.get_shape().ndims == 3:  # [batch_size, w, h]
            if is_mean:
                mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, target), [1, 2]))
            else:
                mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output, target), [1, 2]))
        elif output.get_shape().ndims == 4:  # [batch_size, w, h, c]
            if is_mean:
                mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, target), [1, 2, 3]))
            else:
                mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output, target), [1, 2, 3]))
        else:
            raise Exception("Unknow dimension")
        return mse

def normalized_mean_square_error(output, target):
    with tf.name_scope("normalized_mean_square_error"):
        if output.get_shape().ndims == 2:  # [batch_size, n_feature]
            nmse_a = tf.sqrt(tf.reduce_sum(tf.squared_difference(output, target), axis=1))
            nmse_b = tf.sqrt(tf.reduce_sum(tf.square(target), axis=1))
        elif output.get_shape().ndims == 3:  # [batch_size, w, h]
            nmse_a = tf.sqrt(tf.reduce_sum(tf.squared_difference(output, target), axis=[1, 2]))
            nmse_b = tf.sqrt(tf.reduce_sum(tf.square(target), axis=[1, 2]))
        elif output.get_shape().ndims == 4:  # [batch_size, w, h, c]
            nmse_a = tf.sqrt(tf.reduce_sum(tf.squared_difference(output, target), axis=[1, 2, 3]))
            nmse_b = tf.sqrt(tf.reduce_sum(tf.square(target), axis=[1, 2, 3]))
        nmse = tf.reduce_mean(nmse_a / nmse_b)
    return nmse

def absolute_difference_error(output, target, is_mean=False):
    #Return the TensorFlow expression of absolute difference error (L1) of two batch of data.
   
    with tf.name_scope("absolute_difference_error"):
        if output.get_shape().ndims == 2:  # [batch_size, n_feature]
            if is_mean:
                loss = tf.reduce_mean(tf.reduce_mean(tf.abs(output - target), 1))
            else:
                loss = tf.reduce_mean(tf.reduce_sum(tf.abs(output - target), 1))
        elif output.get_shape().ndims == 3:  # [batch_size, w, h]
            if is_mean:
                loss = tf.reduce_mean(tf.reduce_mean(tf.abs(output - target), [1, 2]))
            else:
                loss = tf.reduce_mean(tf.reduce_sum(tf.abs(output - target), [1, 2]))
        elif output.get_shape().ndims == 4:  # [batch_size, w, h, c]
            if is_mean:
                loss = tf.reduce_mean(tf.reduce_mean(tf.abs(output - target), [1, 2, 3]))
            else:
                loss = tf.reduce_mean(tf.reduce_sum(tf.abs(output - target), [1, 2, 3]))
        else:
            raise Exception("Unknow dimension")
        return loss

def dice_coe(output, target, loss_type='jaccard', axis=[1, 2, 3], smooth=1e-5):
    
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice)
    return dice

def dice_hard_coe(output, target, threshold=0.5, axis=[1, 2, 3], smooth=1e-5):
    """Non-differentiable Sørensen–Dice coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation i.e. labels are binary.
    The coefficient between 0 to 1, 1 if totally match.
    """
    output = tf.cast(output > threshold, dtype=tf.float32)
    target = tf.cast(target > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(output, target), axis=axis)
    l = tf.reduce_sum(output, axis=axis)
    r = tf.reduce_sum(target, axis=axis)

    hard_dice = (2. * inse + smooth) / (l + r + smooth)
    hard_dice = tf.reduce_mean(hard_dice)
    return hard_dice

def iou_coe(output, target, threshold=0.5, axis=[1, 2, 3], smooth=1e-5):
    """Non-differentiable Intersection over Union (IoU) for comparing the
    similarity of two batch of data, usually be used for evaluating binary image segmentation.
    The coefficient between 0 to 1, 1 means totally match.
    """
    pre = tf.cast(output > threshold, dtype=tf.float32)
    truth = tf.cast(target > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(pre, truth), axis=axis)  # AND
    union = tf.reduce_sum(tf.cast(tf.add(pre, truth) >= 1, dtype=tf.float32), axis=axis)  # OR

    batch_iou = (inse + smooth) / (union + smooth)
    iou = tf.reduce_mean(batch_iou)
    return iou  #, pre, truth, inse, union


