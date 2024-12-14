import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, LSTM, 
    Dense, Dropout, Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ModelCheckpoint
)
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class ThoughtToTextModel:
    def __init__(self, input_shape=(64, 250, 1), num_classes=4):
        """
        Initialize CNN-LSTM model for motor imagery classification
        
        Args:
            input_shape (tuple): Input data shape
            num_classes (int): Number of motor imagery classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_cnn_lstm_model()
        self.xgboost_model = None
    
    def _build_cnn_lstm_model(self):
        """
        Construct CNN-LSTM architecture
        
        Returns:
            tf.keras.Model: Compiled neural network model
        """
        model = Sequential([
            # 2D Convolutional Layers for Spatial Features
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            # Flatten and Reshape for LSTM
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            
            # Reshape for LSTM
            tf.keras.layers.RepeatVector(5),  # Adjust timesteps as needed
            
            # LSTM for Temporal Features
            LSTM(64, return_sequences=True),
            LSTM(32),
            
            # Output Layer
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """
        Train the CNN-LSTM model
        
        Args:
            X_train (np.ndarray): Training data
            y_train (np.ndarray): Training labels
            X_val (np.ndarray): Validation data
            y_val (np.ndarray): Validation labels
        """
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )
        
        model_checkpoint = ModelCheckpoint(
            'best_model.h5', 
            save_best_only=True
        )
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stop, model_checkpoint]
        )
        
        return history
    
    def train_xgboost_classifier(self, X_train, y_train):
        """
        Train XGBoost classifier for ensemble/validation
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
        """
        # Flatten and prepare data
        X_flat = X_train.reshape(X_train.shape[0], -1)
        
        self.xgboost_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
        
        self.xgboost_model.fit(X_flat, np.argmax(y_train, axis=1))
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test (np.ndarray): Test data
            y_test (np.ndarray): Test labels
        
        Returns:
            dict: Performance metrics
        """
        # CNN-LSTM Evaluation
        cnn_lstm_pred = self.model.predict(X_test)
        cnn_lstm_accuracy = accuracy_score(
            np.argmax(y_test, axis=1), 
            np.argmax(cnn_lstm_pred, axis=1)
        )
        
        # XGBoost Evaluation
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        xgb_pred = self.xgboost_model.predict(X_test_flat)
        xgb_accuracy = accuracy_score(
            np.argmax(y_test, axis=1), 
            xgb_pred
        )
        
        return {
            'cnn_lstm_accuracy': cnn_lstm_accuracy,
            'xgboost_accuracy': xgb_accuracy,
            'detailed_report': classification_report(
                np.argmax(y_test, axis=1), 
                np.argmax(cnn_lstm_pred, axis=1)
            )
        }
    
    def save_models(self, cnn_path='cnn_lstm_model.pkl', xgb_path='xgboost_model.pkl'):
        """
        Save trained models
        
        Args:
            cnn_path (str): Path to save CNN-LSTM model
            xgb_path (str): Path to save XGBoost model
        """
        self.model.save(cnn_path)
        
        if self.xgboost_model:
            import joblib
            joblib.dump(self.xgboost_model, xgb_path)
    
    def predict(self, input_signal):
        """
        Predict motor imagery class
        
        Args:
            input_signal (np.ndarray): Preprocessed EEG signal
        
        Returns:
            np.ndarray: Prediction probabilities
        """
        # Ensure input is in correct shape
        input_signal = input_signal.reshape(
            (1,) + self.input_shape
        )
        
        return self.model.predict(input_signal)

# Example Usage
def main():
    # Simulated data (replace with actual preprocessed EEG data)
    np.random.seed(42)
    X = np.random.rand(100, 64, 250, 1)  # 100 samples
    y = tf.keras.utils.to_categorical(
        np.random.randint(0, 4, size=(100,)), 
        num_classes=4
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Initialize and train model
    thought_model = ThoughtToTextModel()
    
    # Train CNN-LSTM
    history = thought_model.train_model(
        X_train, y_train, X_val, y_val
    )
    
    # Train XGBoost Classifier
    thought_model.train_xgboost_classifier(X_train, y_train)
    
    # Evaluate models
    results = thought_model.evaluate_model(X_test, y_test)
    print("Model Performance:")
    print(results['detailed_report'])
    
    # Save models
    thought_model.save_models()

if __name__ == "__main__":
    main()