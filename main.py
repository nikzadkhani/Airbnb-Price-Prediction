import numpy as np
import pandas as pd

def process_data(data, train, test, predict, normalize="STD"):
  """ Preprocess the data by mapping all the data to unique integers, then
  normalizes the data. Furthermore, it drops the columns with a large
  amount of missing data and it also drops the rows with missing data.
  :param data: pandas dataframe containing all the data
  :param train: pandas dataframe containing id's of training data
  :param test: pandas dataframe containing id's of testing data
  :param predict: pandas dataframe containing id's of prices we must predict
  :param normalize: Which method of normalization to use.
  """
  print("DATA PREPROCESSING BEGINNING\n")
  assert (normalize == "STD" or normalize == "MAX/MIN")

  # Make Replacement dictionaries
  property_type_d = {"Aparthotel": 0, "Apartment": 1, "Barn": 2, "Bed and breakfast": 3,\
    "Boat": 4, "Boutique hotel": 5, "Bungalow": 6, "Bus": 7, "Cabin": 8,\
    "Camper/RV": 9, "Campsite": 10, "Chalet": 11, "Condominium": 12,\
    "Cottage": 13, "Dome house": 14, "Farm stay": 15,"Guest suite": 16,\
    "Guesthouse": 17, "Hostel": 18, "Hotel": 19, "House": 20, "Houseboat": 21,\
    "Hut": 22, "Loft": 23, "Other": 24, "Resort": 25, "Serviced apartment": 26,\
    "Tent": 27, "Tiny house": 28, "Tipi": 29, "Townhouse": 30, "Treehouse": 31,\
    "Villa": 32, "Yurt": 33}


  response_time_d = {"": 0, "a few days or more": 1, "within a day": 2,\
          "within a few hours": 3 , "within an hour": 4}

  boolean_d = {"t": 1, "f": 0}

  room_type_d = {"Entire home/apt": 0, "Hotel room": 1, "Private room": 2, \
                 "Shared room": 3}

  bed_type_d = {"Airbed": 0, "Couch": 1, "Futon": 2, "Pull-out Sofa": 3, \
                "Real Bed": 4}

  big_d = dict(property_type_d, **response_time_d, **boolean_d, **room_type_d, \
            **bed_type_d)

  # Converts all strings to an integer mapping
  data = data.replace(big_d)
  data = data.replace(to_replace="TX 78702", value="78702")
  data = data.replace(to_replace={'%':''}, regex=True)
  data = data.fillna(-1)
  data["zipcode"] = pd.to_numeric(data["zipcode"])

  data["amenities"] = data["amenities"].apply(len)

  # Converts columns with dates to UNIX timestamps
  date_columns = ["host_since", "first_review", "last_review"]
  for col in date_columns:
    data[col] = pd.to_datetime(data[col])
    data[col] = pd.to_numeric(data[col])

  # Takes out price and id before data normalization
  price = data["price"]
  del data["price"]

  ids = data["id"]
  del data["id"]

  def normalize_max_min(df):
    df = df.astype(float)
    return (df-df.mean())/df.std()

  def normalize_std(df):
    df = df.astype(float)
    return (df-df.mean())/df.std()

  if normalize == "STD":
    data = normalize_std(data)
  else:
    data = normalize_max_min(data)

  # Adds back price and ids
  data = pd.concat((ids, data, price), axis=1)

  # Split up data
  def partition(data, subsetdf):
    subset_indices = list(subsetdf.values.flatten().astype(int))
    return data[data["id"].isin(subset_indices)]

  train = partition(data, train)
  test = partition(data, test)
  predict = partition(data, predict)

  # Save prices
  train_Y = train["price"].values
  test_Y = test["price"].values

  # Delete prices
  del train["price"]
  del test["price"]
  del predict["price"]

  # Delete Ids
  del train["id"]
  del test["id"]
  del predict["id"]

  # Save Ids
  train_X = train.values
  test_X = test.values
  predict_X = predict.values

  return train_X, train_Y, test_X, test_Y, predict_X, data


def h(X, theta):
  return X @ theta.T

def add_column(X):
  num_rows = X.shape[0]
  return np.insert(X, 0, np.ones((1, num_rows)), axis=1)

def mse(X, Y, theta):
  m = X.shape[0]
  return (1/(2 * m)) * ((h(X, theta) - Y).T @ (h(X, theta) - Y))

def mse_grad(X, Y, theta):
  m = X.shape[0]
  grad = X.T @ X @ theta.T
  grad = grad - (X.T @ Y)
  return grad/m

def rmse(X, Y, theta):
  return mse(X, Y, theta) ** 0.5

def rmse_grad(X, Y, theta):
  return 0.5 * rmse(X, Y, theta) @ mse_grad(X, Y, theta).T

def grad_descent(X, Y, test_X, test_Y, num_trials=10000, lr=1e-3, \
                 display_interval=5000, display_thetas=False):
  print("TRAINING BEGINNING\n")
  X = add_column(X)
  test_X = add_column(test_X)
  m, n = X.shape # m = rows, n = columns
  theta = np.zeros((1, n))
  Y = Y.reshape(-1, 1)
  test_Y = test_Y.reshape(-1, 1)

  print("X: ", X.shape)
  print("Y: ", Y.shape)
  print("test_X: ", test_X.shape)
  print("test_Y: ", test_Y.shape)
  print("theta: ", theta.shape)
  print()

  losses = [0.0] * num_trials
  test_losses = [0.0] * num_trials
  thetas = theta

  for i in range(num_trials):
    loss = rmse(X, Y, theta)
    losses[i] = loss

    test_loss = rmse(test_X, test_Y, theta)
    test_losses[i] = test_loss

    theta = theta - (lr * rmse_grad(X, Y, theta))
    thetas = np.concatenate((thetas, theta), axis=0)

    if i % display_interval == 0:
      print("i : ", i, "Training Loss : ", loss, "Test Loss : ", test_loss)

      if display_thetas:
        print(theta)

      print()

    # if i % 8000 == 0:
    #   lr *= 1.1

  thetas = thetas[:-1]

  print("TRAINING COMPLETE\n\n")

  return losses, test_losses, thetas, theta

def predict_and_make_csv(theta, predict_X, predict, decimals=2, filename="Answers.csv"):
  """ Writes and calcs prediction to a csv file witht filename equal to filename
  :param theta: np array of trained theta values
  :param predict_X: np array of X values
  :param predict: pd dataframe with ids of prediction
  :param decimals: number of decimals to round the answers to
  :param filename: the name of the file to write to
  """
  print("WRITE CSV BEGINNING\n")
  np_prices = np.round(h(add_column(predict_X), theta), decimals=decimals)
  prices = pd.DataFrame(np_prices, columns=["price"])
  answers = pd.concat((predict, prices), axis=1)
  answers.to_csv(filename, index=False)
  print("Number of printed answers: ", len(answers))
  print("\nWRITE CSV COMPLETE")

def main():
    with open("./data/data.csv") as f, open("./data/train.csv") as t, \
         open("./data/test.csv") as tst, open("./data/val.csv") as v:
      raw_data = pd.read_csv(f)
      train = pd.read_csv(t)
      test = pd.read_csv(v)
      predict = pd.read_csv(tst)

    X, Y, test_X, test_Y, predict_X, data = \
      process_data(raw_data, train, test, predict)
    print("Training Data: ", len(X))
    print("Testing Data: ", len(test_X))
    print("Prediction Data: ", len(predict))
    print("Total Amount of Data: ", len(data))
    print("\nDATA PREPROCESSING COMPLETE\n\n")

    losses, test_losses, thetas, theta = \
        grad_descent(X, Y, test_X, test_Y, num_trials=20000, lr=3e-3)

    predict_and_make_csv(theta, predict_X, predict, filename="pred.csv")

if __name__ == '__main__':
    main()
