import pandas as pd
import numpy as np
from finrl.finrl_meta.preprocessor.preprocessors import FeatureEngineer
import tensorflow as tf
from keras import layers
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from utils import read_yaml
import scipy
from scipy import signal
from sklearn.metrics import confusion_matrix
import os

class SenarioClassifier:
    def __init__(self, config, is_train):
        self.data_path = config["VNINDEX_PATH"]
        self.data_dir = config["DATA_DIR"]
        self.tech_indicator_list = ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma']
        self.observation_day = config["OBSERVATION_DATE"]
        self.day_of_interest = config["DAY_OF_INTEREST"]
        self.checkpoint_dir = config["CHECKPOINT_DIR"]
        self.threshold = config["THRESHOLD"]
        #Create model
        self.model = Sequential()
        self.model.add(layers.BatchNormalization(axis = -1))
        self.model.add(layers.LSTM(50, input_shape=(self.day_of_interest, 9 + len(self.tech_indicator_list))))
        self.model.add(layers.Dense(1, activation='sigmoid'))
        
        if is_train == False:
            self.model.load_weights(self.checkpoint_dir)
            self.VNIndex = pd.read_csv(os.path.join(self.data_dir,"VNIndex_processed.csv"))
        else:
            self.caculate_indicator(self.tech_indicator_list)
            self.VNIndex.to_csv(os.path.join(self.data_dir,"VNIndex_processed.csv"))

    def caculate_indicator(self,indicator_list):
        VNIndex = pd.read_csv(self.data_path)
        VNIndex.columns = ['tic', 'date', 'open', 'high','low','close','volume']

        VNIndex['date'] = pd.to_datetime(VNIndex['date'].astype(str), format='%Y%m%d')
        VNIndex['date'] = VNIndex['date'].astype(str)

        fe = FeatureEngineer(
                            use_technical_indicator=True,
                            tech_indicator_list = self.tech_indicator_list,
                            use_turbulence=True,
                            use_vix=True,
                            user_defined_feature = False)
        self.VNIndex = fe.preprocess_data(VNIndex)

    def create_dataset(self, X, y, time_steps=30, step=1):
        Xs, ys = [], []
        for i in range(time_steps, len(X), step):
            #v = X.iloc[i-time_steps:i].values
            v = X[i-time_steps:i]
            labels = y.iloc[time_steps]
            Xs.append(v)
        ys = np.where(y.to_numpy() == "Good", 1,0)
        return np.array(Xs), np.array(ys)[time_steps:]

    def train(self, observation_day = 15, day_of_interest = 30):
        
        label = []
        for i in range(len(self.VNIndex)):
            begin_price = self.VNIndex.iloc[max(0, i - day_of_interest)]['close']
            #begin_price = data.iloc[i]['close']
            end_price = self.VNIndex.iloc[min(len(self.VNIndex)-1, i + day_of_interest)]['close']
            if end_price >= begin_price:
                label.append(1)
            else:
                label.append(0)
        #Filter
        label = signal.medfilt(np.array(label,dtype=float), kernel_size= 15)
        label = np.where(label == 1, "Good", "Bad")
        self.VNIndex['label'] = label
        #fig = px.scatter(self.VNIndex, x="date", y="close", color="label")

        X, y = self.create_dataset(self.VNIndex[['close','open','high','low','volume','macd','macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma', 'vix', 'turbulence']].to_numpy(), self.VNIndex['label'], 30)
        index = list(range(0,len(X), observation_day))
        X = np.array([X[i] for i in index])
        y = np.array([y[i] for i in index])
        print(X.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'], )

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_dir,
                                                        save_weights_only=True,
                                                        verbose=1,
                                                        mode='max',
                                                        monitor='val_accuracy',
                                                        save_best_only=True)


        history = self.model.fit(X_train, y_train, epochs=1000, batch_size=64, validation_data = (X_test, y_test), callbacks=[cp_callback])
        
        self.model.load_weights(self.checkpoint_dir)
        # Final evaluation of the model
        y_pred = self.model.predict(X_test)
        scores = self.model.evaluate(X_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))
    
        #Get confuse matrix
        cf_matrix = confusion_matrix(y_test, y_pred > self.threshold)
        print(cf_matrix)

    def predict(self):
        VNI_value = self.VNIndex.iloc[-15:][['close','open','high','low','volume','macd','macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma', 'vix', 'turbulence']].to_numpy().astype(np.float64)
        scores = self.model.predict(np.expand_dims(VNI_value, axis=0))[0][0]
        print("scores:", scores)
        if scores >= self.threshold:
            print("Good - Probability = ", scores * 100)
            return 1, scores * 100
        else:
            print("Bad - Probability = ", (1-scores) * 100)
            return 0, (1-scores) * 100

if __name__ == "__main__":
    config_path = "config_LP.yaml"
    config = read_yaml(config_path)
    classifier = SenarioClassifier(config, is_train=True)
    classifier.train()