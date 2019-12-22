using CSV
import Random: shuffle

function string_to_number(x::Array)
    placeholder = []
    for item in x
        if typeof(item) == String
            push!(placeholder, parse(Float32, item))
        else
            push!(placeholder, item)
        end
    end
    return placeholder
end

function load_data(path, batch_size, start; _shuffle=true)
    x = []
    y = []
    pairs = []
    for row in CSV.Rows(path, skipto=start, limit=batch_size)
        rows = [row.V1, row.V2, row.V3, row.V4, row.V5, row.V6, row.V7, row.V8, row.V9,
                row.V10, row.V11, row.V12, row.V13, row.V14, row.V15, row.V16, row.V17,
                row.V18, row.V19, row.V20, row.V21, row.V22, row.V23, row.V24, row.V25,
                row.V26, row.V27, row.V28, log(parse(Float32, row.Amount)+0.0001)]
        rows = string_to_number(rows)
        push!(pairs, rows => parse(Float32, row.Class))
    end

    if _shuffle
        pairs = shuffle(pairs)
    end

    for pair in pairs
        push!(x, pair.first)
        push!(y, pair.second)
    end
    return gpu(gpu.(x)), gpu(gpu.(y))
end

function neg_pos(file)
    neg = 0
    pos = 0
    for row in CSV.Rows(file)
        if parse(Float32, row.Class) == 0
            neg += 1
        else
            pos += 1
        end
    end
    return neg, pos
end
