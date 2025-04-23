# using Pkg ; Pkg.add("HypothesisTests"); Pkg.add("StatsBase");Pkg.add("ScikitLearnBase");Pkg.add("GaussianMixtures");Pkg.add("ShiftedArrays");
# using Pkg;Pkg.add("CSV");Pkg.add("DataFrames");Pkg.add("Statistics");
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
    freq = 60 
    n_diff_of_gmm = 7

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
    if !flag || any(isnan, prediction)  # Add NaN check for predictions
        return L(Inf)
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

    # 1. Test if Log returns is normal distribution 
    diffs_z = (diffs .- mean(diffs)) ./ sigma
    dist = Normal(0, 1)  # Theoretical normal distribution
    ks_test = ApproximateOneSampleKSTest(diffs_z, dist)
    stats = ks_test.δ

    
    # Add validation for statistics
    if isnan(stats) 
        return L(Inf)
    end
    
    normality_stats = (10^(stats))

    # 10.5 is by experimental. pvalue is reversed by the pkg, so p_sw nears 0 is normally distribution .
    if normality_stats>1.04
        return L(normality_stats)
    end 

    try 
        # GMM Regression 
        shift_diffs = ShiftedArrays.lag(diffs, n_diff_of_gmm)
        shift_diffs = Float64.(replace(shift_diffs, missing => NaN))

        d = hcat(diffs, shift_diffs)
        d = d[(n_diff_of_gmm+1):end,:]

        if any(isnan, d) || size(d, 1) < 2
            return L(normality_stats)
        end

        gmm_model = GaussianMixtures.GMM(2, 2, kind=:diag)  
        gmm_model = ScikitLearnBase.fit!(gmm_model, d)
        pred_label = ScikitLearnBase.predict(gmm_model, d)

        if maximum(pred_label) == minimum(pred_label)
            return L(normality_stats)
        end 

        pred_label = ShiftedArrays.lag(pred_label,1) 
        pred_label = Float64.(replace(pred_label, missing => 0.0))
        pred_label = vec(pred_label)
        
        d_col1 = d[:, 1]
        d_col1 = Float64.(replace(d_col1, missing => 0.0))

        d_col1_label_1 = d_col1[pred_label .== 1]
        d_col1_label_0 = d_col1[pred_label .== 2]

        # Add validation for empty groups
        if isempty(d_col1_label_1) || isempty(d_col1_label_0)
            return L(normality_stats)
        end

        sum_d_col_1 = sum(d_col1_label_1)
        sum_d_col_0 = sum(d_col1_label_0)

        net_val = abs(sum_d_col_0) + abs(sum_d_col_1) + 1

        loss = 1/net_val
        
        println("&&&&& Success, net_val: ", (net_val/n)*(250*6*60), ", normality_stats: ", normality_stats, " &&&&&")
        return L(loss)
    catch err 
        return L(normality_stats)
    end 
end
