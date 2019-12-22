using CSV

dataset = CSV.read("dataset/creditcard.csv")
test_data_size = round(Int, data_size*0.2)

train_data = dataset[1:end-test_data_size, :]
test_data = dataset[end-test_data_size+1:end, :]

CSV.write("dataset/train_data.csv", train_data)
CSV.write("dataset/test_data.csv", test_data)
