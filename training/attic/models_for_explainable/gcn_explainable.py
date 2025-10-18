import tensorflow as tf
from spektral.layers import GCNConv, GlobalMaxPool
from tensorflow.keras.layers import Dense, Dropout

class ExplainableGCN(tf.keras.Model):
    def __init__(self, num_features, num_labels, hidden_units, dense_units, dropout_rate):
        super(ExplainableGCN, self).__init__()
        self.conv1 = GCNConv(hidden_units, activation='relu')
        self.conv2 = GCNConv(hidden_units, activation='relu')
        self.conv3 = GCNConv(hidden_units, activation='relu')
        self.global_pool = GlobalMaxPool()

        self.dropout1 = Dropout(dropout_rate) # Dodat dropout
        self.dropout2 = Dropout(dropout_rate) # Dodat dropout

        self.dense1 = Dense(dense_units[0], activation='relu', 
                            kernel_regularizer=tf.keras.regularizers.l2(0.01))

        self.dense2 = Dense(dense_units[1], activation='relu', 
                            kernel_regularizer=tf.keras.regularizers.l2(0.01))

        self.output_layer = Dense(num_labels, activation='softmax')

    def call(self, inputs, training=False):
        x, a = inputs 
        
        # GNN slojevi sa Dropoutom
        x = self.conv1([x, a])
        x = self.dropout1(x, training=training)

        x = self.conv2([x, a])
        x = self.dropout2(x, training=training)
        
        #x = self.conv3([x, a]) #za regularan trening
        last_conv_output = self.conv3([x, a]) #za explainable deo
        
        # Korišćenje GlobalMaxPool umesto tf.reduce_mean
        pooled_output = self.global_pool(last_conv_output)
        
        x = self.dense1(pooled_output)
        x = self.dense2(x)
        output = self.output_layer(x)

        return last_conv_output, output