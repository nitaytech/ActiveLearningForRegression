# TODO: add description to each variable

DATA_FILE = 'data/milk_sessions.csv'
RESULTS_FOLDER = 'results'

DATE_COLUMN = 'Date'

FEATURES = ['DailyYield_KG', 'DailyFat_P', 'DailyProtein_P', 'DailyConductivity', 'DailyActivity',
            'CurrentRP', 'CurrentMET', 'CurrentKET', 'CurrentMF', 'CurrentPRO', 'CurrentLDA', 'CurrentMAST',
            'CurrentEdma', 'CurrentLAME', 'Disease', 'DIM', 'DIM_50-175', 'DIM_<50', 'DIM_>=175',
            'LactationNumber', 'Fertility_Num', 'Age', 'Still', 'Milk', 'MilkTemperature', 'Fat',
            'Protein', 'Lactose', 'LogScc', 'Cf', 'Blood', 'Casein', 'Mufa', 'Pufa', 'Sfa', 'Ufa',
            'Pa', 'Sa', 'Oa']


Y_COLUMN = 'Component7',

G_FEATURES = ['Fat', 'Protein', 'CaseinMeanCalibrated', 'DIM_50-175', 'Lactose',]
B_FEATURES = ['Disease', 'DIM_<50', 'DIM_>=175', 'LogScc', 'DailyConductivity']


N_INIT = 2
N_INSTANCES = 2
INIT_STEPS = [5, 10, 15, 20, 25]
TEST_SIZE = 0.05
N_CLUSTERS = 10 * N_INSTANCES
BINS = 8
RANDOM_SEED = 42


