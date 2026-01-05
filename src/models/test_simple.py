# test_simple.py
try:
    import tensorflow as tf
    print("✅ TensorFlow installed successfully!")
    print("Version:", tf.__version__)
    
    # تحميل بيانات بسيطة
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    print("✅ CIFAR-10 data loaded!")
    print("Training data shape:", x_train.shape)
    
except ImportError:
    print("❌ TensorFlow not installed!")