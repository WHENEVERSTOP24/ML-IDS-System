import os

# Project base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Dataset URLs (using HoaNP's verified repository)
TRAIN_URL = "https://raw.githubusercontent.com/HoaNP/NSL-KDD-DataSet/master/KDDTrain+.txt"
TEST_URL = "https://raw.githubusercontent.com/HoaNP/NSL-KDD-DataSet/master/KDDTest+.txt"

# Dataset raw and processed file paths
RAW_TRAIN_PATH = os.path.join(DATA_DIR, "KDDTrain+.txt")
RAW_TEST_PATH = os.path.join(DATA_DIR, "KDDTest+.txt")

PROCESSED_TRAIN_PATH = os.path.join(DATA_DIR, "train_processed.csv")
PROCESSED_TEST_PATH = os.path.join(DATA_DIR, "test_processed.csv")

# Full column list of NSL-KDD raw files
# The raw dataset has 43 columns: 41 features, 1 target label (attack type), 1 difficulty level
COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty_level'
]

# Categorical and numerical column separations
CATEGORICAL_FEATURES = ['protocol_type', 'service', 'flag']

# Binary mapping
# Target column labels: 'normal' vs 'attack' (we will group all other labels as attack)
BINARY_TARGET = 'label_binary'
# Multi-class mapping (normal, dos, probe, r2l, u2r)
MULTICLASS_TARGET = 'label_multiclass'

# Attack mappings to broad categories
ATTACK_MAPPING = {
    'normal': 'normal',
    
    # DoS (Denial of Service)
    'back': 'dos',
    'land': 'dos',
    'neptune': 'dos',
    'pod': 'dos',
    'smurf': 'dos',
    'teardrop': 'dos',
    'mailbomb': 'dos',
    'apache2': 'dos',
    'processtable': 'dos',
    'udpstorm': 'dos',
    'worm': 'dos',
    
    # Probe (Probing / Scanning)
    'satan': 'probe',
    'ipsweep': 'probe',
    'portsweep': 'probe',
    'nmap': 'probe',
    'mscan': 'probe',
    'saint': 'probe',
    
    # R2L (Remote to Local)
    'guess_passwd': 'r2l',
    'ftp_write': 'r2l',
    'imap': 'r2l',
    'phf': 'r2l',
    'spy': 'r2l',
    'multihop': 'r2l',
    'warezmaster': 'r2l',
    'warezclient': 'r2l',
    'snmpgetattack': 'r2l',
    'snmpguess': 'r2l',
    'xsnoop': 'r2l',
    'xlock': 'r2l',
    'sendmail': 'r2l',
    'named': 'r2l',
    'httptunnel': 'r2l',
    
    # U2R (User to Root)
    'buffer_overflow': 'u2r',
    'loadmodule': 'u2r',
    'rootkit': 'u2r',
    'perl': 'u2r',
    'sqlattack': 'u2r',
    'xterm': 'u2r',
    'ps': 'u2r'
}
