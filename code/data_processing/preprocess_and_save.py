import numpy as np
import nina_funcs as nf
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#base_dir = "C:/Users/PC/Desktop/TCC_Folders/TCC_code/TCC_molina/code/Databases/NinaproDB2/"
base_dir = r"C:\Users\PC\Downloads\DB5/"
output_dir = "E:/db5"
os.makedirs(output_dir, exist_ok=True)


train_reps = [1, 3, 4, 6]
test_reps = [2, 5]
win_len = 400
win_stride = 20


gestures_dict = {
    "1": [1,2,3,4,5,6,7,8,9,10],
    "2": [1,2,3,4,5,6,7,8,9,10]
}

def preprocess_and_save_data(subject_id, exercise_id, folder_path, file_name, gestures):
    data = nf.get_data(folder_path, file_name)
    
    data = nf.normalise(data, train_reps)
    emg_band = nf.filter_data(data=data, f=(20, 500), butterworth_order=4, btype='bandpass')
    
    X_train, y_train, _ = nf.windowing(emg_band, train_reps, gestures, win_len, win_stride)
    X_test, y_test, _ = nf.windowing(emg_band, test_reps, gestures, win_len, win_stride)
    
    y_train = nf.get_categorical(y_train)
    y_test = nf.get_categorical(y_test)
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        y_train = np.argmax(y_train, axis=1)
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_test = np.argmax(y_test, axis=1)
    
    np.save(os.path.join(output_dir, f"S{subject_id}E{exercise_id}_X_train.npy"), X_train)
    np.save(os.path.join(output_dir, f"S{subject_id}E{exercise_id}_y_train.npy"), y_train)
    np.save(os.path.join(output_dir, f"S{subject_id}E{exercise_id}_X_test.npy"), X_test)
    np.save(os.path.join(output_dir, f"S{subject_id}E{exercise_id}_y_test.npy"), y_test)
    print(f"Processamento concluído para sujeito {subject_id}, exercício {exercise_id}")

def process_individual_files():
    tasks = []
    with ProcessPoolExecutor() as executor:
        for sub_dir in os.listdir(base_dir):
            full_sub_dir = os.path.join(base_dir, sub_dir)
            if os.path.isdir(full_sub_dir): 
                for file in os.listdir(full_sub_dir):
                    if file.endswith(".mat"):
                        parts = file.split('_')
                        subject_id = parts[0][1:]
                        exercise_id = parts[1][1]
                        
                        if exercise_id in gestures_dict and exercise_id in ["1", "2"]:
                            gestures = gestures_dict[exercise_id]
                            print(f"Enviando sujeito {subject_id}, exercício {exercise_id} para processamento...")
                            tasks.append(
                                executor.submit(preprocess_and_save_data, subject_id, exercise_id, full_sub_dir, file, gestures)
                            )
        
        for future in as_completed(tasks):
            future.result()  

if __name__ == '__main__':
    print("Iniciando processamento paralelo...")
    process_individual_files()
    print("Processamento paralelo concluído e arquivos salvos.")
