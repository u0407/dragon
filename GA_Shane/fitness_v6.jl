# using Pkg ; Pkg.add("HypothesisTests"); Pkg.add("StatsBase");Pkg.add("ScikitLearnBase");Pkg.add("GaussianMixtures");Pkg.add("ShiftedArrays");
# using Pkg;Pkg.add("CSV");Pkg.add("DataFrames");Pkg.add("Statistics");
import Pkg; Pkg.add("Distributions");
import Statistics
using StatsBase
using HypothesisTests
using Distributions
using ScikitLearnBase # 兼容 Scikit-Learn 风格接口
using GaussianMixtures
using ShiftedArrays
using CSV
using DataFrames


function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    # Target # of output rows in minute freq of 27 minutes
    freq = 15 

    # Ensure constants are int with no digits . 
    for node in tree
        if node.degree == 0 && node.constant
            val = node.val::T
            if isnan(val)
                return L(Inf)
            end
            node.val = convert(T, round(Int, val))
        end
    end

    # predict with X 
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    prediction2, flag2 = eval_tree_array(tree, dataset.X*100, options)

    if !flag || any(isnan, prediction)  # Add NaN check for predictions
        return L(Inf)
    end

    if (prediction[100] != prediction2[100]) && (prediction[10] != prediction2[10])
        multiplier = 3
    else
        multiplier = 1
    end

    # Predicted operators
    p = abs.(prediction)
    if any(isnan, p) || any(isinf, p) || all(x -> x ≈ 0, p)
        return L(Inf)
    end

    a0_vals = dataset.y
    n = length(p)
    
    A0_lst = []
    # !!!! Thred Calculation 
    thred = Statistics.mean(p) * freq
    
    if isnan(thred) 
        return L(Inf)
    end
    # !!!!  Bar definition 
    cumsum = 0.0
    for i in 1:n
        cumsum += p[i]
        if cumsum > thred || i == n
            push!(A0_lst, a0_vals[i])
            cumsum = 0.0
        end
    end

    # Ensure the number of bars 
    if length(A0_lst) < (length(a0_vals) / (freq*1.5))
        return L(Inf)
    end 
        
    # Returns of A0_lst
    diffs = diff(A0_lst)
    diffs = filter(!isnan, diffs)

    if isempty(diffs)
        return L(Inf)
    end

    sigma = std(diffs)
    if isnan(sigma)
        return L(Inf)
    end

    diffs_z = (diffs .- mean(diffs)) ./ sigma
    
    # dist = Normal(0, 1)  
    # ks_test = ApproximateOneSampleKSTest(diffs_z, dist)
    # stats = ks_test.δ

    dist = Normal(0, 1)  
    ad_test = OneSampleADTest(diffs_z, dist)
    stats = ad_test.A² 
    stats = stats/multiplier
    # jb_test= JarqueBeraTest(diffs_z)
    # stats = 1/jb_test.JB * 100

    # println(" stats:", stats, " kurt: ", kurtosis(diffs_z) )
    if isnan(stats)
        return L(Inf)
    end
    
    stats = log(stats)
    normality_stats = (2^stats)
    return L(normality_stats)

end

