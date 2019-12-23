import BSON: @load
using Plots, DelimitedFiles

@load "checkpoints/model_epoch-1_batch-23.bson" local_model

local_model = gpu(local_model)

const test_file = "dataset/test_data.csv"
const test_data_size = 56961

x_test, y_test = load_data(test_file, test_data_size, 2)
test_acc = accuracy(x_test, y_test, local_model) # 0.9388199261506147 0.911

loss_data = readdlm("losses.txt")
accuracy_data = readdlm("accuracy.txt")

x_loss = loss_data[:,1]
y_loss = loss_data[:,2]

loss_plot = plot(y_loss, x_loss,
                label="",
                title="Training Loss",
                xlabel="Time_step")

x_acc = accuracy_data[:,1]
y_acc = accuracy_data[:,2]

accuracy_plot = plot(y_acc, x_acc, label="",
                title="Training Accuracy",
                xlabel="Time_step")

savefig(loss_plot, "loss_plot")
savefig(accuracy_plot, "accuracy_plot")
