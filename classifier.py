import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
import os
import pickle
import argparse
import sys
warnings.filterwarnings('ignore')

def extract_segment_features(segment, sr, n_mfcc=13):
    """
    Extract features from an audio segment
    """
    try:
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
        
        # Extract additional features
        spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(segment)
        
        # Compute statistics
        features = []
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))
        features.append(np.mean(spectral_centroid))
        features.append(np.std(spectral_centroid))
        features.append(np.mean(spectral_rolloff))
        features.append(np.std(spectral_rolloff))
        features.append(np.mean(zero_crossing_rate))
        features.append(np.std(zero_crossing_rate))
        
        return np.array(features)
    except:
        return None

def segment_audio(audio_path, segment_duration=1.0, sr=22050):
    """
    Split audio file into segments
    """
    y, sr = librosa.load(audio_path, sr=sr)
    segment_samples = int(segment_duration * sr)
    
    segments = []
    for i in range(0, len(y), segment_samples):
        segment = y[i:i + segment_samples]
        if len(segment) >= segment_samples * 0.5:
            segments.append(segment)
    
    return segments, sr

def load_training_data(train_folder, sr=22050):
    """
    Load all audio files from train/cat and train/dog folders
    """
    print("Loading training data...")
    
    X_train = []
    y_train = []
    
    # Load cat sounds (label = 0)
    cat_folder = os.path.join(train_folder, 'cat')
    if not os.path.isdir(cat_folder):
        print(f"Warning: Cat folder '{cat_folder}' not found. Skipping cats.")
        cat_files = []
    else:
        cat_files = [f for f in os.listdir(cat_folder) if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))]

    print(f"Found {len(cat_files)} cat audio files")
    for filename in cat_files:
        filepath = os.path.join(cat_folder, filename)
        try:
            segments, _ = segment_audio(filepath, sr=sr)
            for segment in segments:
                features = extract_segment_features(segment, sr)
                if features is not None:
                    X_train.append(features)
                    y_train.append(0)  # 0 for cat
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Load dog sounds (label = 1)
    dog_folder = os.path.join(train_folder, 'dog')
    if not os.path.isdir(dog_folder):
        print(f"Warning: Dog folder '{dog_folder}' not found. Skipping dogs.")
        dog_files = []
    else:
        dog_files = [f for f in os.listdir(dog_folder) if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))]

    print(f"Found {len(dog_files)} dog audio files")
    for filename in dog_files:
        filepath = os.path.join(dog_folder, filename)
        try:
            segments, _ = segment_audio(filepath, sr=sr)
            for segment in segments:
                features = extract_segment_features(segment, sr)
                if features is not None:
                    X_train.append(features)
                    y_train.append(1)  # 1 for dog
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"\nTotal training segments: {len(X_train)}")
    print(f"Cat segments: {np.sum(y_train == 0)}")
    print(f"Dog segments: {np.sum(y_train == 1)}")
    
    return X_train, y_train, sr

def train_model(train_folder, model_path='cat_dog_classifier.pkl'):
    """
    Train the classifier on all files in train/cat and train/dog folders
    """
    print("=== TRAINING PHASE ===\n")
    
    # Load training data
    X_train, y_train, sr = load_training_data(train_folder)
    
    if len(X_train) == 0:
        raise ValueError("No training data found!")
    
    # Train Random Forest classifier
    print("\nTraining classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    clf.fit(X_train, y_train)
    
    # Evaluate on training data
    y_pred = clf.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)
    print(f"Training accuracy: {accuracy:.2%}")
    
    # Save the model
    model_data = {
        'classifier': clf,
        'sr': sr
    }
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel saved to: {model_path}")
    
    return clf, sr

def test_model(test_folder, model_path='cat_dog_classifier.pkl'):
    """
    Test the trained model on files in test/cat and test/dog folders
    """
    print("\n=== TESTING PHASE ===\n")
    
    # Load the trained model
    print(f"Loading model from: {model_path}")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    clf = model_data['classifier']
    sr = model_data['sr']
    
    # Test on cat folder
    print("\n--- Testing on Cat Files ---")
    cat_folder = os.path.join(test_folder, 'cat')
    cat_results = test_folder_files(cat_folder, clf, sr, true_label='cat')
    
    # Test on dog folder
    print("\n--- Testing on Dog Files ---")
    dog_folder = os.path.join(test_folder, 'dog')
    dog_results = test_folder_files(dog_folder, clf, sr, true_label='dog')
    
    # Overall summary
    print("\n" + "="*60)
    print("=== OVERALL TEST RESULTS ===")
    
    total_cat_detected = cat_results['cat_count'] + dog_results['cat_count']
    total_dog_detected = cat_results['dog_count'] + dog_results['dog_count']
    
    print(f"\nTotal segments classified:")
    print(f"  Cat sounds detected: {total_cat_detected}")
    print(f"  Dog sounds detected: {total_dog_detected}")
    
    # Accuracy calculation
    correct_predictions = cat_results['cat_count'] + dog_results['dog_count']
    total_predictions = cat_results['total'] + dog_results['total']
    
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"\nTest Accuracy: {accuracy:.2f}%")
        
        print(f"\nBreakdown:")
        print(f"  Cat files: {cat_results['cat_count']}/{cat_results['total']} correctly classified ({cat_results['cat_count']/cat_results['total']*100:.1f}%)")
        print(f"  Dog files: {dog_results['dog_count']}/{dog_results['total']} correctly classified ({dog_results['dog_count']/dog_results['total']*100:.1f}%)")

def test_folder_files(folder_path, classifier, sr, true_label):
    """
    Test all audio files in a folder
    """
    if not os.path.isdir(folder_path):
        print(f"Warning: Folder '{folder_path}' not found. Skipping {true_label} files.")
        return {'cat_count': 0, 'dog_count': 0, 'total': 0}

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))]

    print(f"Found {len(files)} files in {folder_path}")
    
    cat_count = 0
    dog_count = 0
    total_segments = 0
    
    for filename in files:
        filepath = os.path.join(folder_path, filename)
        print(f"\nProcessing: {filename}")
        
        try:
            segments, _ = segment_audio(filepath, sr=sr)
            
            file_cat = 0
            file_dog = 0
            
            for segment in segments:
                features = extract_segment_features(segment, sr)
                if features is not None:
                    features = features.reshape(1, -1)
                    prediction = classifier.predict(features)[0]
                    
                    if prediction == 0:
                        file_cat += 1
                    else:
                        file_dog += 1
            
            cat_count += file_cat
            dog_count += file_dog
            total_segments += (file_cat + file_dog)
            
            print(f"  Segments: Cat={file_cat}, Dog={file_dog}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\nSummary for {true_label} files:")
    print(f"  Total segments: {total_segments}")
    print(f"  Classified as cat: {cat_count}")
    print(f"  Classified as dog: {dog_count}")
    
    return {
        'cat_count': cat_count,
        'dog_count': dog_count,
        'total': total_segments
    }

def main():
    parser = argparse.ArgumentParser(description="Cat vs Dog Sound Classifier")
    parser.add_argument('--data-root', default='cats_dogs', help='Root folder containing train/ and test/ subfolders')
    parser.add_argument('--model-path', default='cat_dog_classifier.pkl', help='Path to save/load model')
    parser.add_argument('--train-only', action='store_true', help='Only train')
    parser.add_argument('--test-only', action='store_true', help='Only test')
    args = parser.parse_args()

    data_root = args.data_root
    train_folder = os.path.join(data_root, 'train')
    test_folder = os.path.join(data_root, 'test')
    model_path = args.model_path

    print("=== Cat vs Dog Sound Classifier ===\n")

    if not os.path.isdir(data_root):
        print(f"Error: Data root folder '{data_root}' not found!")
        print("Expected structure:")
        print("  cats_dogs/train/cat/")
        print("  cats_dogs/train/dog/")
        print("  cats_dogs/test/cat/")
        print("  cats_dogs/test/dog/")
        return

    if args.test_only and not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found for testing.")
        return

    # Train the model (unless --test-only)
    if not args.test_only:
        try:
            clf, sr = train_model(train_folder, model_path)
        except Exception as e:
            print(f"Training failed: {e}")
            return

    # Test the model (unless --train-only)
    if not args.train_only:
        try:
            test_model(test_folder, model_path)
        except Exception as e:
            print(f"Testing failed: {e}")
            return

# If you want to train only
def train_only():
    train_folder = os.path.join('cats_dogs', 'train')
    model_path = "cat_dog_classifier.pkl"
    clf, sr = train_model(train_folder, model_path)

# If you want to test only (using previously trained model)
def test_only():
    test_folder = os.path.join('cats_dogs', 'test')
    model_path = "cat_dog_classifier.pkl"
    test_model(test_folder, model_path)

def classify_single_file(audio_file, model_path='cat_dog_classifier.pkl'):
    """
    Classify a single audio file as cat or dog
    
    Args:
        audio_file: path to the audio file
        model_path: path to the saved model (default: 'cat_dog_classifier.pkl')
    
    Returns:
        'cat' or 'dog'
    """
    # Load the trained model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Please train the model first.")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    clf = model_data['classifier']
    sr = model_data['sr']
    
    # Check if file exists
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    
    # print(f"Classifying: {audio_file}")
    
    # Segment the audio
    segments, _ = segment_audio(audio_file, sr=sr)
    
    cat_votes = 0
    dog_votes = 0
    
    # Classify each segment
    for segment in segments:
        features = extract_segment_features(segment, sr)
        if features is not None:
            features = features.reshape(1, -1)
            prediction = clf.predict(features)[0]
            
            if prediction == 0:
                cat_votes += 1
            else:
                dog_votes += 1
    
    # Determine final classification by majority vote
    if cat_votes > dog_votes:
        result = "cat"
        confidence = (cat_votes / (cat_votes + dog_votes)) * 100
    else:
        result = "dog"
        confidence = (dog_votes / (cat_votes + dog_votes)) * 100
    
    return result

if __name__ == "__main__":
    # main()
    result = classify_single_file("cat_3.wav")
    if(result == 'dog'):
        print("yyyaaaaahhh its a dog ✨🐶")
    else:
        print("its a cat 💢")