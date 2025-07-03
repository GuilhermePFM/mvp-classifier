# symmetric encryption
from cryptography.fernet import Fernet
import pandas as pd 


# Define Encrypt User Defined Function 
def encrypt_val(clear_text,MASTER_KEY):
    f = Fernet(MASTER_KEY)
    clear_text_b=bytes(str(clear_text), 'utf-8')
    cipher_text = f.encrypt(clear_text_b)
    cipher_text = str(cipher_text.decode('ascii'))
    return cipher_text


def decrypt_val(cipher_text,MASTER_KEY):
    f = Fernet(MASTER_KEY)
    clear_val=f.decrypt(cipher_text.encode()).decode()
    return clear_val

def generate_key():
    key = Fernet.generate_key()
    with open('symmetric_key.key', 'wb') as f:
        f.write(key)
    return key

def encrypt_file(file_path,MASTER_KEY):
    with open(file_path, 'rb') as f:
        file_data = f.read()
    
    fernet = Fernet(MASTER_KEY)
    encrypted_data = fernet.encrypt(file_data)
    with open(file_path, 'wb') as f:
        f.write(encrypted_data)

def read_decrypted_file(file_path,MASTER_KEY):
    with open(file_path, 'rb') as f:
        file_data = f.read()
        
    fernet = Fernet(MASTER_KEY)
    decrypted_data = fernet.decrypt(file_data)
    return decrypted_data

def encrypt_dataset(dataset:pd.DataFrame, MASTER_KEY):
    encrypted_dataset = dataset.copy()
    for column in encrypted_dataset.columns:
        encrypted_dataset[column] = encrypted_dataset[column].apply(lambda x: encrypt_val(str(x),MASTER_KEY))
    return encrypted_dataset, dataset.dtypes


def decrypt_dataset(encrypted_dataset:pd.DataFrame,MASTER_KEY, dtypes:dict):
    decrypted_dataset = encrypted_dataset.copy()
    for column in decrypted_dataset.columns:
        decrypted_dataset[column] = decrypted_dataset[column].apply(lambda x: decrypt_val(x,MASTER_KEY))

    return decrypted_dataset.astype(dtypes)