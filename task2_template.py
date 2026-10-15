import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import IterableDataset, TensorDataset
from torch.utils.data import DataLoader
import math
import sys
import os
import time
import csv
import numpy as np
import sqlite3
import json
import optuna

# Total user count: 138493
# Max user idx: 138492
# Total item count: 26744
# Max item idx: 26743

# Number of unique users in train: 138493
# Number of unique items in train: 25876
# Number of unique users in test: 138307
# Number of unique items in test: 20261

# no duplicate rows in train and test
# Number of users in test but not in train: 0
# Number of items in test but not in train: 868

# Max Timestamp from Ratings_Train: 1,427,784,002
# Min Timestamp from Ratings_Train: 789,652,004
# Max Timestamp from Ratings_Test: 1,427,782,288
# Min Timestamp from Ratings_Test: 822,873,600

# RatingsCV
# fold_id | count
# (0, 3,255,554)
# (1, 3,227,798)
# (2, 3,199,971)
# (3, 3,172,356)
# (4, 3,144,532)

# RatingsTuning
# fold_id | count
# (0, 1,174,648)
# (1, 1,164,667)
# (2, 1,154,659)
# (3, 1,144,656)
# (4, 1,134,640)
NUM_WORKERS = 6
BATCH_SIZE = 4096

# Tuned Hyperparameters
FEATURES = 16
LEARNING_RATE = 0.005
cur_db = 'user_item_ratings.db'

class MFModel(nn.Module):
    """Matrix Factorization Model"""
    def __init__(self, num_users: int, num_items: int, features: int, global_mean: float, max_norm=1.0) -> None:
        """Initializes model parameters"""
        super().__init__()

        self.user_embeddings = nn.Embedding(num_embeddings=num_users, embedding_dim=features,sparse=True, max_norm=max_norm)
        self.item_embeddings = nn.Embedding(num_embeddings=num_items, embedding_dim=features, sparse=True, max_norm=max_norm)

        # Biases
        self.user_bias = nn.Embedding(num_users, 1, sparse=True)
        self.item_bias = nn.Embedding(num_items, 1, sparse=True)

        # Uniform distribution of initial weights
        self.user_embeddings.weight.data.normal_(0, 0.01)
        self.item_embeddings.weight.data.normal_(0, 0.01)
        self.user_bias.weight.data.zero_()
        self.item_bias.weight.data.zero_()

        self.global_mean = global_mean

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass"""

        user_embeddings = self.user_embeddings(users)
        item_embeddings = self.item_embeddings(items)

        # dot product
        out = (user_embeddings * item_embeddings).sum(dim=1)
        # add bias
        out = out + self.user_bias(users).squeeze() + self.item_bias(items).squeeze()
        # add global mean
        return out + self.global_mean

class SQLiteDataset(IterableDataset):
    def __init__(self, table_name:str, mode="train", val_fold=0,
                 db=cur_db, row_chunk = 10000) -> None:
        if not table_exists(table_name, db):
            raise ValueError(f"{table_name} does not exist")
        if mode not in ["train", "val", "test"]:
            raise ValueError("mode must be 'train', 'val', or 'test'")

        self.table_name = table_name
        self.mode = mode
        self.val_fold = val_fold
        self.db = db
        self.row_chunk = row_chunk

    def __iter__(self):
        connection = sqlite3.connect(self.db)
        cursor = connection.cursor()

        # Optimization commands:
        cursor.execute("PRAGMA journal_mode = WAL")  # Faster concurrent reading
        cursor.execute("PRAGMA cache_size = -10000")  # Use 10MB of RAM for cache
        cursor.execute("PRAGMA synchronous = OFF")

        worker_info = torch.utils.data.get_worker_info()

        if self.mode == 'test':
            select = "user_idx, item_idx"
        else:
            select = "user_idx, item_idx, rating"

        if self.mode == "train":
            # Exclude validation fold data
            sql_condition = "WHERE fold_id != ?"
            params = [self.val_fold]
        elif self.mode == "val":
            # ONLY include validation fold data
            sql_condition = "WHERE fold_id = ?"
            params = [self.val_fold]
        else:
            # Test set doesn't care about folds, grab everything
            sql_condition = "WHERE is_train = 0"
            params = []

        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            # Append the modulo constraint to shard rows across your CPU cores
            sql_condition += " AND (rowid % ?) = ?"
            params.extend([num_workers, worker_id])

        # Execute the final dynamically built query
        query = f"SELECT {select} FROM {self.table_name} {sql_condition}"
        cursor.execute(query, tuple(params))

        while True:
            rows = cursor.fetchmany(self.row_chunk)
            if not rows:
                break
            if not self.mode == "test":
                for user, item, rating in rows:
                    yield torch.tensor(user), torch.tensor(item), torch.tensor(rating, dtype=torch.float32)
            else:
                for user, item in rows:
                    yield torch.tensor(user), torch.tensor(item)

        cursor.execute("PRAGMA synchronous = NORMAL")
        connection.close()

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, path="best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, current_loss, model):
        if current_loss < (self.best_loss - self.min_delta):
            self.best_loss = current_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# Table Creation
def csv_helper(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            yield row[:-1] # create a new list without timestamp

def create_main_table(db=cur_db):
    # define connection and cursor
    connection = sqlite3.connect(db)
    cursor = connection.cursor()

    # create ratings table from training data
    create_ratings_table = """CREATE TABLE IF NOT EXISTS 
    Ratings (
        idx INTEGER PRIMARY KEY,
        user_id INTEGER NOT NULL, 
        item_id INTEGER NOT NULL, 
        rating FLOAT, 
        is_train BOOLEAN NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_rating_user_id ON Ratings (user_id);
    CREATE INDEX IF NOT EXISTS idx_rating_item_id ON Ratings (item_id);""" # faster to create index after insertion
    cursor.executescript(create_ratings_table)

    print("Inserting train data")
    start_time = time.time()
    insert_train_data = "INSERT INTO Ratings (user_id, item_id, rating, is_train) VALUES (?,?,?,1)"
    cursor.executemany(insert_train_data, csv_helper('train_20M.csv'))
    print("Insertion completed in " + str(time.time() - start_time) + " seconds")

    select_first_ten_train = "SELECT * FROM Ratings WHERE is_train = 1 LIMIT 10"
    train_rows = cursor.execute(select_first_ten_train).fetchall()
    print("TRAIN DATA")
    for row in train_rows:
        print(row)
    print()

    print("Inserting test data")
    start_time = time.time()
    insert_test_data = "INSERT INTO Ratings (user_id, item_id, rating, is_train) VALUES (?,?,NULL, 0)"
    cursor.executemany(insert_test_data, csv_helper('test_20M.csv'))
    print("Insertion completed in " + str(time.time() - start_time) + " seconds")

    select_first_ten_test = "SELECT * FROM Ratings WHERE is_train = 0 LIMIT 10"
    test_rows = cursor.execute(select_first_ten_test).fetchall()
    print("TEST DATA")
    for row in test_rows:
        print(row)
    print()

    connection.commit()
    connection.close()

def create_mapping_tables(db=cur_db):
    connection = sqlite3.connect(db)
    cursor = connection.cursor()

    create_mapping_script = """
    --Create User Mapping Table
    CREATE TABLE IF NOT EXISTS 
    UserMapping (
        user_idx INTEGER PRIMARY KEY,
        user_id INTEGER NOT NULL UNIQUE 
    );
    
    INSERT INTO UserMapping (user_idx, user_id)
    SELECT (ROW_NUMBER() OVER (ORDER BY user_id) - 1), user_id
    FROM (SELECT DISTINCT user_id FROM Ratings) AS u;

    -- Create Item Mapping Table
    CREATE TABLE ItemMapping (
        item_idx INTEGER PRIMARY KEY,
        item_id INTEGER NOT NULL UNIQUE
    );
    
    INSERT INTO ItemMapping (item_idx, item_id)
    SELECT (ROW_NUMBER() OVER (ORDER BY item_id) - 1), item_id
    FROM (SELECT DISTINCT item_id FROM Ratings) AS i;"""
    cursor.executescript(create_mapping_script)

    connection.commit()
    connection.close()

def create_indexed_table(db=cur_db):
    connection = sqlite3.connect(db)
    cursor = connection.cursor()

    create_index_script = """
    CREATE TABLE IF NOT EXISTS
    RatingsIndex (
        row_id INTEGER PRIMARY KEY,
        user_idx INTEGER NOT NULL,
        item_idx INTEGER NOT NULL,
        rating FLOAT,
        is_train BOOLEAN NOT NULL
    );
    
    --Bulk Insert Data
    PRAGMA synchronous = OFF;
    PRAGMA journal_mode = MEMORY;
    INSERT INTO RatingsIndex (user_idx, item_idx, rating, is_train)
    SELECT u.user_idx, i.item_idx, r.rating, r.is_train
    FROM Ratings r
    JOIN UserMapping u ON r.user_id = u.user_id
    JOIN ItemMapping i ON r.item_id = i.item_id;
    
    CREATE INDEX idx_indexed_lookup_cv ON RatingsIndex (is_train, user_idx);
    CREATE UNIQUE INDEX idx_indexed_lookup ON RatingsIndex (is_train, user_idx, item_idx);
    
    PRAGMA journal_mode = WAL;
    PRAGMA synchronous = NORMAL;
    ANALYZE RatingsIndex;"""
    cursor.executescript(create_index_script)

    connection.commit()
    connection.close()

def create_cv_table(db=cur_db):
    connection = sqlite3.connect(db)
    cursor = connection.cursor()

    create_cv_script = """CREATE TABLE IF NOT EXISTS
    RatingsCV AS
    SELECT 
        row_id, user_idx, item_idx, rating, 
        (ROW_NUMBER() OVER( PARTITION BY user_idx ORDER BY RANDOM()) - 1) % 5 AS fold_id
    FROM RatingsIndex WHERE is_train = 1;
    
    CREATE UNIQUE INDEX idx_cv_fold_train ON RatingsCV (fold_id, user_idx, item_idx);
    ANALYZE RatingsCV;"""
    cursor.executescript(create_cv_script)

    connection.commit()
    connection.close()

def create_tuning_table(db=cur_db):
    connection = sqlite3.connect(db)
    cursor = connection.cursor()

    create_tuning_script = """CREATE TABLE IF NOT EXISTS
    RatingsTuning (
        row_id INTEGER PRIMARY KEY,
        user_idx INTEGER NOT NULL,
        item_idx INTEGER NOT NULL,
        rating FLOAT,
        fold_id INTEGER NOT NULL
    );
    
    --Bulk Insert Data
    PRAGMA synchronous = OFF;
    PRAGMA journal_mode = MEMORY;
    INSERT INTO RatingsTuning
    SELECT row_id, user_idx, item_idx, rating, fold_id
    FROM RatingsCV
    WHERE user_idx IN (
        SELECT user_idx FROM UserMapping ORDER BY RANDOM() LIMIT 5000
    );
    
    --Create Indexes
    CREATE UNIQUE INDEX idx_tuning_user_item ON RatingsTuning (user_idx, item_idx);
    CREATE INDEX idx_tuning_fold ON RatingsTuning (fold_id);
    
    PRAGMA journal_mode = WAL;
    PRAGMA synchronous = NORMAL;
    ANALYZE RatingsTuning;
    """
    cursor.executescript(create_tuning_script)

    connection.commit()
    connection.close()

# SQLite Functions
def table_exists(table: str, db=cur_db):
    connection = sqlite3.connect(db)
    cursor = connection.cursor()

    # Check the master table
    cursor.execute('''
        SELECT count(name) FROM sqlite_master 
        WHERE type='table' AND name=?
    ''', (table,))

    # fetchone() returns a tuple like (1,) or (0,)
    exists = cursor.fetchone()[0] == 1

    connection.close()
    return exists

def get_query_plan(query:str, db=cur_db):
    connection = sqlite3.connect(db)
    cursor = connection.cursor()

    cursor.execute(f"EXPLAIN QUERY PLAN {query}")
    plan = cursor.fetchall()
    for row in plan:
        print(row)

    connection.close()

def analyze_tables(db=cur_db):
    connection = sqlite3.connect(db)
    cursor = connection.cursor()

    cursor.execute("ANALYZE;")

    connection.commit()
    connection.close()

def select_sql_script(script:str, db=cur_db):
    connection = sqlite3.connect(db)
    cursor = connection.cursor()
    cursor.execute(script)
    res = cursor.fetchall()
    connection.commit()
    connection.close()
    return res

def idx_to_id(id_type:str, table: str, idx_arr, db=cur_db):
    if id_type not in ['user', 'item']:
        raise TypeError("id_type not user or item")
    if table not in['UserMapping', 'ItemMapping']:
        raise TypeError("table not accepted")
    connection = sqlite3.connect(db)
    cursor = connection.cursor()

    idx_list = [int(x) for x in idx_arr]
    placeholders = ",".join(["?"] * len(idx_list))
    query = f"SELECT {id_type}_idx, {id_type}_id FROM {table} WHERE {id_type}_idx IN ({placeholders})"

    cursor.execute(query, idx_list)
    result_dict = dict(cursor.fetchall())

    connection.close()

    user_ids = [result_dict[idx] for idx in idx_list if idx in result_dict]
    return user_ids

# Data analysis
def print_num_unique_ids(id_col:str, table:str, db=cur_db):
    connection = sqlite3.connect(db)
    cursor = connection.cursor()

    select_distinct = f"SELECT COUNT (DISTINCT {id_col}) FROM {table}"
    res = cursor.execute(select_distinct).fetchone()[0]
    print(f"Number of unique {id_col} in {table}: {res}")

    connection.close()

def print_non_train_elem(db=cur_db):
    connection = sqlite3.connect(db)
    cursor = connection.cursor()

    select_test_users_not_in_train = "SELECT user_id FROM Ratings_Test EXCEPT SELECT user_id FROM Ratings_Train"
    num = len(cursor.execute(select_test_users_not_in_train).fetchall())
    print(f"Number of users in test but not in train: {num}")

    select_test_items_not_train = "SELECT item_id FROM Ratings_Test EXCEPT SELECT item_id FROM Ratings_Train"
    num = len(cursor.execute(select_test_items_not_train).fetchall())
    print(f"Number of items in test but not in train: {num}")
    connection.close()

def get_max_id(id_col:str, table:str, db=cur_db):
    connection = sqlite3.connect(db)
    cursor = connection.cursor()

    res = cursor.execute(f"SELECT MAX({id_col}) FROM {table}").fetchall()[0][0]
    connection.close()

    return res

# Matrix Factorization helper functions
def get_total_count(table:str, db=cur_db):
    connection = sqlite3.connect(db)
    cursor = connection.cursor()

    res = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchall()[0][0]
    connection.close()

    return res

def get_global_mean(db=cur_db):

    connection = sqlite3.connect(db)
    cursor = connection.cursor()

    res = cursor.execute("SELECT AVG(rating) FROM RatingsCV").fetchall()[0][0]
    connection.close()

    return res

def get_fold_means_and_counts(table:str, db=cur_db):
    if table == "RatingsCV":
        cached_file = 'full_fold_stats.json'
    elif table == "RatingsTuning":
        cached_file = 'tune_fold_stats.json'
    else:
        raise ValueError("Table must either be RatingsCV or RatingsTuning")

    if os.path.exists(cached_file):
        with open(cached_file, 'r') as f:
            data = json.load(f)
            return {int(k): v for k, v in data.items()}

    connection = sqlite3.connect(db)
    cursor = connection.cursor()

    select_fold_means_and_counts = f"""
    SELECT
        fold_id,
        AVG(rating) as fold_mean,
        COUNT(*) as row_count
    FROM {table}
    GROUP BY fold_id"""

    results = cursor.execute(select_fold_means_and_counts).fetchall()
    stats = {row[0]: {'mean': row[1], 'count':row[2]} for row in results}

    with open(cached_file, 'w') as f:
        json.dump(stats, f)

    return stats

def calculate_train_mean(fold_stats, validation_fold:int):
    if validation_fold < 0 or validation_fold > 4:
        raise ValueError("Validation fold must be between 0 and 4")
    total_sum = 0.0
    total_count = 0

    for f_id, stats in fold_stats.items():
        if f_id == validation_fold:
            continue # skip validation fold

        total_sum += stats['mean'] * stats['count']
        total_count += stats['count']

    return total_sum / total_count if total_count != 0 else 0

# Matrix Factorization Main Functions
def train_one_epoch(model, train_loader: DataLoader,
                    optimizer, loss_function, device, epoch,
                    print_batches=True, batch_report=10):
    model.train()

    # initialize total loss variable
    total_loss = 0
    count = 0

    # nested for loop to implement Stochastic Gradient Descent / loss function
    for batch_idx, (users, items, ratings) in enumerate(train_loader):
        # Move to GPU if available
        users, items, ratings = users.to(device), items.to(device), ratings.to(device)

        # Forward pass
        predictions = model(users, items)

        # get loss by comparing predicted value against the ground truth
        loss = loss_function(predictions, ratings)

        # Clear gradients
        optimizer.zero_grad()

        # Back propagation (to compute gradients)
        loss.backward()

        # Update parameters
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()
        count += 1

        # Progress every x batches
        if print_batches and batch_idx % batch_report == 0:
            print(f"Epoch {epoch} | Batch {batch_idx} | Loss {loss.item()}")

    # print progress every epoch
    avg_loss = total_loss / count
    return avg_loss

def evaluate_model(model, val_loader: DataLoader, device):
    model.eval()

    total_absolute_error = 0
    total_squared_error = 0
    total_count = 0

    with torch.no_grad():
        for users, items, ratings in val_loader:
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)

            preds = model(users, items)
            preds = torch.clamp(preds, 0.5, 5.0)

            batch_absolute_error = torch.abs(preds - ratings)
            batch_squared_error = torch.pow(preds - ratings, 2)

            total_absolute_error += batch_absolute_error.sum()
            total_squared_error += batch_squared_error.sum()
            total_count += ratings.size(0)

    mean_absolute_error = total_absolute_error / total_count
    mean_squared_error = total_squared_error / total_count
    root_mean_squared_error = math.sqrt(mean_squared_error)

    return {"mae": mean_absolute_error, "mse": mean_squared_error, "rmse": root_mean_squared_error}

def predict_ratings(model, test_loader: DataLoader, device, file='result.csv'):
    model.eval()

    with open(file, mode='w', newline='') as f:
        writer = csv.writer(f, delimiter=',')

        with torch.no_grad():
            for users, items in test_loader:
                users, items = users.to(device), items.to(device)

                preds = model(users, items)
                preds = torch.clamp(preds, 0.5, 5.0)

                hard_preds = torch.round(preds * 2) / 2 # 0.5 increments

                users = users.cpu().numpy().flatten()
                items = items.cpu().numpy().flatten()

                user_ids = idx_to_id('user', "UserMapping", users)
                item_ids = idx_to_id("item", "ItemMapping", items)

                hard_preds = hard_preds.cpu().numpy().flatten()

                writer.writerows(zip(user_ids, item_ids, hard_preds))

def trainModel(train_dataset, val_dataset, global_mean: float,
               val_metric='mae', learning_rate = 0.001, num_epochs = 10,
               features = 2, max_norm=1.0, batch_size=BATCH_SIZE, save_file:str=None,
               print_epochs=True, epoch_report=5,
               print_batches=True, batch_report=10):
    if val_metric not in ['mae', 'mse', 'rmse']:
        raise ValueError("val_metric must be either mae, mse or rmse")

    # Initialize model
    print("Initializing Model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_users = get_total_count("UserMapping")
    num_items = get_total_count("ItemMapping")
    model = MFModel(
        num_users=num_users,
        num_items=num_items,
        features=features, global_mean=global_mean, max_norm=max_norm)
    model.to(device)

    model_config = {
        "num_users": num_users,
        "num_items": num_items,
        "features": features,
        "global_mean": global_mean,
        "max_norm": max_norm
    }

    # Define loss function, for this example we will be using Mean Squared Error
    loss_function = nn.MSELoss()

    # Define optimizer (Stochastic Gradient Descent) with a learning rate of 0.01
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=learning_rate)

    # Initialize Loaders for Training and Validation
    print("Start Loading")
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size,
        num_workers=NUM_WORKERS, pin_memory=False)
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=batch_size,
        num_workers=NUM_WORKERS, pin_memory=False)

    # Initialize EarlyStopper
    stopper = EarlyStopping(patience=5,min_delta=0.001)

    print("Starting Training")
    model.train()
    metric = 0
    # Training iteration
    for epoch in range(1,num_epochs+1):
        avg_train_loss = train_one_epoch(model, train_loader, optimizer,
                                         loss_function, device, epoch,
                                         print_batches=print_batches, batch_report=batch_report)

        metrics = evaluate_model(model, val_loader, device)
        if print_epochs and epoch % epoch_report == 0:
            print("=" * 100)
            print(f"Epoch {epoch} | Average Loss: {avg_train_loss:.6f}")
            print(f"Validation MAE: {metrics['mae']:.6f} | MSE: {metrics['mse']:.6f} | RMSE: {metrics['rmse']:.6f}")
            print("=" * 100)

        metric = metrics[val_metric]
        stopper(metrics[val_metric], model)
        if stopper.early_stop:
            print(f"Stopping early. Loading best weights with loss of {stopper.best_loss:.6f}")
            if metrics[val_metric] < stopper.best_loss:
                model.load_state_dict(torch.load(stopper.path))
                metric = stopper.best_loss
            break

    if save_file:
        model_info = {
            "model_config": model_config,
            "state_dict": model.state_dict(),
        }
        torch.save(model_info, save_file)
    return model, metric


def model_sanity_check():
    print("getting means")
    fold_means_and_counts = get_fold_means_and_counts("RatingsTuning")
    print("calculating means")
    mean = calculate_train_mean(fold_means_and_counts, 0)

    print("Initializing Train Dataset")
    train_dataset = SQLiteDataset(
        table_name="RatingsTuning", mode="train", row_chunk=10000)
    val_dataset = SQLiteDataset(
        table_name="RatingsTuning", mode="val", row_chunk=10000)

    trainModel(
        train_dataset, val_dataset,
        global_mean=mean, num_epochs=2, features=2, batch_size=BATCH_SIZE,
        print_epochs=True, epoch_report=1, print_batches=False
    )

def create_objective(table:str, cv_num=2):
    if table not in ['RatingsCV', 'RatingsTuning']:
        raise ValueError("table must be either RatingsCV or RatingsTuning")

    if cv_num > 5 or cv_num < 1:
        raise ValueError("cv_num must be between 1 and 5")

    def objective(trial):
        # Hyperparameters to Tune
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        features = trial.suggest_categorical("features", [2,4,8,16])
        num_epochs = 3
        max_norm = 1.0

        fold_stats = get_fold_means_and_counts(table)

        scores = []
        for fold_id in range(cv_num):
            train_dataset = SQLiteDataset(table_name=table, mode="train", val_fold=fold_id, row_chunk=50000)
            val_dataset = SQLiteDataset(table_name=table, mode="val", val_fold=fold_id, row_chunk=50000)
            global_mean = calculate_train_mean(fold_stats, fold_id)
            _, mae = trainModel(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                global_mean=global_mean,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                features=features,
                max_norm=max_norm,
                print_epochs=True,
                epoch_report=1,
                print_batches=False,
                batch_report=10
            )
            scores.append(mae)

        return float(np.mean(scores))

    return objective

def run_hyperparameter_tuning(table:str, cv_num=2, n_trials=50):
    objective = create_objective(table, cv_num)

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(),
        storage="sqlite:///optuna.sqlite.db",
        load_if_exists=True
    )

    print("Starting Optuna Study...")
    start = time.time()
    study.optimize(objective, n_trials=n_trials)

    print(f"Finished Optuna Study in {time.time() - start:.2f} seconds")
    print(f"Best MAE Loss: {study.best_value}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

def final_training():
    train_dataset = SQLiteDataset(table_name="RatingsCV", mode="train", val_fold=0)
    val_dataset = SQLiteDataset(table_name="RatingsCV", mode="val", val_fold=0)

    model, metrics = trainModel(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        global_mean=get_global_mean(),
        val_metric='mae',
        learning_rate=LEARNING_RATE,
        num_epochs=3,
        features=FEATURES,
        max_norm=1.0,
        batch_size=BATCH_SIZE,
        save_file='full_model.pth',
        print_epochs=True,
        epoch_report=1,
        print_batches=False,
        batch_report=0
    )

    print(f"Final MAE: {metrics}")

def final_inference():
    model_info = torch.load('full_model.pth', map_location=torch.device('cpu'))

    saved_config = model_info["model_config"]

    model = MFModel(
        num_users=saved_config["num_users"],
        num_items=saved_config["num_items"],
        features=saved_config["features"],
        global_mean=saved_config["global_mean"],
        max_norm=saved_config["max_norm"]
    )

    model.load_state_dict(model_info["state_dict"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    test_dataset = SQLiteDataset(
        table_name="RatingsIndex",
        mode='test'
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    predict_ratings(model, test_loader, device)

if __name__ == "__main__":
    final_inference()
    print("Done!")