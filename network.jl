using Flux, CuArrays
using DelimitedFiles
import Flux: binarycrossentropy, normalise,  crossentropy
import Statistics: mean
import BSON: @save, @load

const file = "dataset/train_data.csv"
const data_size = 227846
batch_size = 2048

last_epoch = 1
last_batch = 0

include("utils.jl")
include("datapreprocessing.jl")

function extract_num(array)
    data = []
    for arr in array
        for num in arr
            push!(data, num)
        end
    end
    return data
end

model = Chain(
        Dense(29, 64, relu),
        Dropout(0.5),
        Dense(64, 1, sigmoid)
) |> gpu
#=
x_test, y_test = load_data(file, batch_size, 2)

x_test_copy = reduce(hcat, x_test)

y_pred_test = reshape(model(x_test_copy), :,1)

data_test = (x_test_copy, y_test)
=#
function loss(x, y)
    sum(binarycrossentropy.(reshape(model(x),:,1), y))
end

function loss_function(x, y)
    if y != 0
        loss = binarycrossentropy(model(x)[1], y) .* 100
    else
        loss = binarycrossentropy(model(x)[1], y) .* 1
    end
    return loss
end

function loss_total(xs, ys)
    losses = []
    ŷs = reduce(vcat, model.(xs))
    for i in zip(ŷs, ys)
        if i[2] != 0
            push!(losses, binarycrossentropy(i[1], i[2]) .* 100)
        else
            push!(losses, binarycrossentropy(i[1], i[2]) .* 1)
        end
    end
    return sum(losses)
end

function accuracy(x, y, _model)
    scores = reduce(vcat, _model.(x))
    fpr, tpr, thresholds = roc_curve(cpu(y), cpu(scores))
    custom_auc(fpr, tpr)
end

opt = ADAM(0.0001)
ps = Flux.params(model)
worst_loss = 50000

function train(epochs::Int64, last_epoch::Int64, file::String, batch_size::Int64, last_batch::Int64)
    batches = round(Int64, ((data_size)/2)/batch_size)
    _last_epoch = last_epoch
    _last_batch = last_batch
    _step = 1

    for epoch = _last_epoch:epochs
        @info("\n- - - - - - - Epoch $epoch - - - - - - -")
        batch = _last_batch
        place = (batch*batch_size) + 2

        for i in place:batch_size:round(Int64, data_size/2)
            batch += 1
            x, y = load_data(file, batch_size, i)

            Flux.train!(loss_function, ps, zip(x,y), opt)

            _loss = loss_total(x, y)
            _acc = accuracy(x, y, model)

            open("losses.txt", "a") do lo
                writedlm(lo, [_loss _step])
            end
            if ~isnan(_acc)
                open("accuracy.txt", "a") do ac
                    writedlm(ac, [_acc _step])
                end
            end

            _step += 1

            if batch == 1 || batch == batches || ~isnan(_acc)
                @info("Epoch: $(epoch) - Batch: [$(batch)/$(batches)] - Sum loss: $(_loss) - Accuracy: $(_acc)")
                if batch == 1 || batch == batches
                    global model
                    local_model = cpu(model)
                    @save "checkpoints/model_epoch-$(epoch)_batch-$(batch).bson" local_model
                end
            end
            if _loss < worst_loss
                global model
                local_model = cpu(model)
                @info("Epoch: $(epoch) - Batch: [$(batch)/$(batches)] - Loss: $(_loss) - Accuracy: $(_acc)")
                @info("New best loss! Saving model...")
                @save "checkpoints/model_epoch-$(epoch)_batch-$(batch).bson" local_model
                global worst_loss = _loss
            end
        end
        _last_batch = 0
        _last_epoch += 1
    end
end

train(5, last_epoch, file, batch_size, last_batch)
