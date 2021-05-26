DATA_FILE = 'data/milk_sessions.csv'
RESULTS_FOLDER = 'results'

DATE_COLUMN = 'Date'
COW_COLUMN = 'CowID'
AL_FEATURES = ['DailyYield_KG', 'DailyFat_P', 'DailyProtein_P', 'DailyConductivity', 'DailyActivity',
               'CurrentRP', 'CurrentMET', 'CurrentKET', 'CurrentMF', 'CurrentPRO', 'CurrentLDA', 'CurrentMAST',
               'CurrentEdma', 'CurrentLAME', 'Disease', 'DIM', 'LactationNumber', 'Fertility_Num',
               'Age', 'Still', 'Milk', 'MilkTemperature', 'Fat', 'Protein', 'Lactose', 'LogScc', 'Cf', 'Blood',
               'Casein', 'Mufa', 'Pufa', 'Sfa', 'Ufa', 'Pa', 'Sa', 'Oa']


AL_Y = 'Component7',

AL_G_FEATURES = ['DailyYield_KG', 'DailyFat_P', 'DailyProtein_P', 'DailyConductivity', 'DailyActivity',
               'DailyRestPerBout', 'Casein', 'Mufa', 'Pufa', 'Sfa', 'Ufa', 'Pa', 'Sa', 'Oa']
AL_B_FEATURES = ['CurrentRP', 'CurrentMET', 'CurrentKET', 'CurrentMF',
               'CurrentPRO', 'CurrentLDA', 'CurrentMAST', 'CurrentEdma', 'CurrentLAME', 'Disease',
               'DIM', 'LactationNumber', 'Fertility_Num', 'Age', 'Still']


AL_TRAIN_DAYS = 10
AL_TEST_DAYS = 14
AL_N_INIT = 2
AL_N_INSTANCES = 2
AL_N_CLUSTERS = 10 * AL_N_INSTANCES
AL_BINS = 8



