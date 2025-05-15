module LoopKit

using TensorKit
using KrylovKit

export coarse_grain_TRG, entanglement_filtering, loop

    """
    Performing the usual Levin-Nave TRG.
    Dcut: dimension of cutoff
    """
    function coarse_grain_TRG(TA, TB, Dcut::Int)
        Tludr = transpose(TA, (1,2),(4,3))
        T1, S1, T2, err = tsvd(Tludr; trunc = truncdim(Dcut))
        println("error = $err")
        Turld = transpose(TB, (2,3), (1,4))
        T3, S2, T4 = tsvd(Turld; trunc = truncdim(Dcut))
        @planar begin
        T1[1 2; md] = T1[1 2; c] * sqrt(S1)[c; md]
        T2[md; 4 3] = sqrt(S1)[md; c] * T2[c; 4 3]
        T3[2 3; md] = T3[2 3; c] * sqrt(S2)[c; md]
        T4[md; 1 4] = sqrt(S2)[md; c] * T4[c; 1 4]
        end
        @planar newT[l u r d] := T2[l; L U] * T4[u; U R] * T3[L D; d] * T1[D R; r]
        return newT
    end



    """
    Perform the entanglement filtering.
    N_ef: total loops for entanglement_filtering.
    epsilon: cutoff when obtaining the projector.

    The leg of the initial local tensor are in the following direction
                |
                ^
                |
                2
                |
     ----<--1---T---3--<----
                |
                4
                |
                ^
                |
    And tensors in a loop is of the order
      |  |
    --B--A--
      |  |
    --A--B--
      |  |
    It is transformed into
      |  |
    --4--1--
      |  |
    --3--2--
      |  |

    """
    function entanglement_filtering(A, B; N_ef::Int = 5, epsilon::Float64 = 1e-12)
        loop = to_T_array(A, B)
        type = eltype(A)
        L = Vector(undef, 4)
        R = Vector(undef, 4)
        for i in 1:4
            L[i] = id(type, space(loop[i],1))
            R[i] = id(type, space(loop[i],4))
        end

        for n = 1:N_ef
            for il in 1:4
                @planar LT[1 2 3 4] := L[il][1; a] * loop[il][a 2 3 4]
                temp = transpose(LT, (1,2,3), (4,))
                _, L[next(il)] = leftorth(temp,)
            end
            L[1] = L[1] / norm(L[1],Inf)
            for ir in reverse(1:4)
                @planar TR[1 2 3 4] := loop[ir][1 2 3 a] * R[ir][4; a]
                temp = transpose(TR, (2,3,4), (1,))
                _, R[last(ir)] = leftorth(temp,)
            end
            R[4] = R[4] / norm(R[4],Inf)
        end

        PR = Vector(undef, 4)
        PL = Vector(undef, 4)

        for i in 1:4
            PR[last(i)], PL[i] = to_projector(L[i], R[last(i)], epsilon)
        end

        @planar newA[1 2 3 4] := A[a b c d] * PL[1][a;1] * PR[3][b;2] * PL[3][c;3] * PR[1][d;4]
        @planar newB[1 2 3 4] := B[a b c d] * PR[2][a;1] * PL[2][b;2] * PR[4][c;3] * PL[4][d;4]

        return newA, newB
    end

    """
    |          |
    ^          v
    |          |
    L          R
    |          |
    \\       /     
      ------
    =
    |           |
    ^           v
    |           |
    U           V
    |           |
    \\        /     
      ---S---   
    """
    function to_projector(L, R, epsilon::Float64)
    @planar LR[a; b] := L[a; c] * R[b; c]
        U, S, V = tsvd(LR; trunc = truncerr(epsilon))
        @planar PR[a; b] := R[c; a] * V'[c;d] * inv(sqrt(S))[d; b]
        @planar PL[a; b] := inv(sqrt(S))[b; d] * L[c; a] * U'[d;c]
        return PR, PL
    end

    next(i::Int) = mod(i,4)+1
    nextS(i::Int) = mod(i,8)+1
    last(i::Int) = mod(i-2,4)+1

    function to_T_site(S_site)
        T_site = 0
        if S_site in (1, 2)
            T_site = 1
        elseif S_site in (3, 4)
            T_site = 2
        elseif S_site in (5, 6)
            T_site = 3
        elseif S_site in (7, 8)
            T_site = 4
        end

        return T_site
    end



    function svd_init(A, B, Dcut::Int)
        transpA = transpose(A, (1,2), (4,3))
        transpB = transpose(B, (2,3), (1,4))
        T1, S1, T2, error1 = tsvd(transpA; trunc = truncdim(Dcut))
        T3, S2, T4, error2 = tsvd(transpB; trunc = truncdim(Dcut))
        println("error1 = $error1, error2 = $error2")
        @planar begin
        T1[1 2; md] = T1[1 2; c] * sqrt(S1)[c; md]
        T2[md; 4 3] = sqrt(S1)[md; c] * T2[c; 4 3]
        T3[2 3; md] = T3[2 3; c] * sqrt(S2)[c; md]
        T4[md; 1 4] = sqrt(S2)[md; c] * T4[c; 1 4]
        end

        return T1, T2, T3, T4
    end

    function to_T_array(A, B)
        loop_T_array = Vector(undef, 4)
        loop_T_array[1] = A
        loop_T_array[2] = transpose(B, (2,3,4,1),())
        loop_T_array[3] = transpose(A, (3,4,1,2),())
        loop_T_array[4] = transpose(B, (4,1,2,3),())
        return loop_T_array
    end

    function to_S_array(loop_T_array, D_cut::Int)
        loop_S_array = Vector(undef, 8)

        for site_T = 1:4
            temp = transpose(loop_T_array[site_T], (1, 2), (4, 3))
            U, S, V, eps = tsvd(temp; trunc = truncdim(D_cut))
            @planar loop_S_array[2*site_T-1][1 2 3] := U[1 2; md] * sqrt(S)[md; 3]
            @planar loop_S_array[2*site_T][1 2 3] := sqrt(S)[1; md] * V[md; 3 2]

        end

        return loop_S_array
    end

    function to_SS_array(loop_S_array)
        loop_SS_array = Vector(undef, 8)
        for i = 1:8
            @planar loop_SS_array[i][ld lu; rd ru] := loop_S_array[i][ld u rd] * loop_S_array[i]'[(); lu u ru]
        end
        return loop_SS_array
    end

    function to_TT_array(loop_T_array)
        loop_TT_array = Vector(undef, 4)
        for i = 1:4
            @planar loop_TT_array[i][ld lu; rd ru] := loop_T_array[i][ld l r rd] * loop_T_array[i]'[(); lu l r ru]
        end
        return loop_TT_array
    end

    function to_TSS_array(loop_T_array, loop_S_array)
        loop_TSS_array = Vector(undef, 4)
        for i = 1:4
            @planar loop_TSS_array[i][ld lu; rd ru] := loop_S_array[2*i-1]'[(); lu l u] * loop_S_array[2*i]'[(); u r ru] * loop_T_array[i][ld l r rd]
        end
        return loop_TSS_array
    end

    function to_number(loop_array)
        cont = loop_array[1]
        for i in 2:length(loop_array)
            @planar T[ld lu; rd ru] := cont[ld lu; md mu] * loop_array[i][md mu; rd ru]
            cont = T
        end

        @planar num = cont[d u; d u]
        return num
    end

    """
    Calculating the cost function.
    """
    function cost_function(loop_TT_array, loop_TSS_array, loop_SS_array)
        numTT = to_number(loop_TT_array)
        numTSS = to_number(loop_TSS_array)
        numSS = to_number(loop_SS_array)
        return (numTT + numSS - numTSS - conj(numTSS))/numTT
    end

    function loop_initialization(A, B, D_cut::Int)
        loop_T = to_T_array(A, B)
        loop_S = to_S_array(loop_T, D_cut)
        loop_TT = to_TT_array(loop_T)
        loop_SS = to_SS_array(loop_S)
        loop_TSS = to_TSS_array(loop_T, loop_S)
        return loop_T, loop_S, loop_TT, loop_SS, loop_TSS
    end

    function to_N(loop_S_array, loop_SS_array, site::Int)
        s = nextS(site)
        TNN = loop_SS_array[s]
        for i = 2:7
            s = nextS(s)
            @planar T[ld lu; rd ru] := TNN[ld lu; md mu] * loop_SS_array[s][md mu; rd ru]
            TNN = T
        end

        return TNN
    end

    function to_W(loop_T_array, loop_S_array, loop_TSS_array, S_site::Int)
        T_site = to_T_site(S_site)

        site = next(T_site)

        TSS = loop_TSS_array[site]
        for i = 1:2
            site = next(site)
            @planar T[ld lu; rd ru] := TSS[ld lu; md mu] * loop_TSS_array[site][md mu; rd ru]
            TSS = T
        end

        if S_site in (2, 4, 6, 8)
            S_site_comp = S_site - 1
            @planar TS[ld lu; rd r ru] := loop_T_array[T_site][ld l r rd] * loop_S_array[S_site_comp]'[(); lu l ru]
            #---4-S†-5---
            #     |   
            #     |   |
            #     |   1
            #      \ /     
            #----3--T--2----
            @planar W[l u r] := TS[ld lu; md u l] * TSS[md r; ld lu]
            
        elseif S_site in (1, 3, 5, 7)
            S_site_comp = S_site + 1
            @planar TS[ld l lu; rd ru] := loop_T_array[T_site][ld l r rd] * loop_S_array[S_site_comp]'[(); lu r ru]
            #    ---4-S†-5--
            #         |   
            #     |   |
            #     3   |
            #      \ /     
            #----2--T--1----
            @planar W[l u r] := TS[ld u r; md mu] * TSS[md mu; ld l]

        end

        return W
    end

    """
    Optimizing a single local S-tensor
    """
    function opt_S(N, W)
        function apply_f(x)
            @planar Npsi[l u r] := N[ld r; rd l] * x[rd u ld]
            return Npsi
        end

        new_S, info = linsolve(apply_f, W)
        return new_S
    end

    """
    The optimization step of Loop-TNR, changing local tensors in a loop such that the cost in a loop is minimized.
    """
    function sweep!(loop_T_array, loop_S_array, loop_TT_array, loop_SS_array, loop_TSS_array, N_sweep::Int, relative_descend::Float64, absolute_error::Float64)
        lastcost = 0.
        for n = 1:N_sweep
            cost = cost_function(loop_TT_array, loop_TSS_array, loop_SS_array)
            ratio = abs(lastcost - cost)/abs(lastcost)
            println("n_sweep: $n, cost = $cost, relative_descend = $ratio")
            if (ratio > relative_descend) && (abs(cost) > absolute_error)
                for S_site = 1:8
                    N = to_N(loop_S_array, loop_SS_array, S_site)
                    W = to_W(loop_T_array, loop_S_array, loop_TSS_array, S_site)
                    loop_S_array[S_site] = opt_S(N, W)
                    @planar SS[ld lu; rd ru] := loop_S_array[S_site][ld u rd] * loop_S_array[S_site]'[(); lu u ru]
                    loop_SS_array[S_site] = SS
                    T_site = to_T_site(S_site)
                    @planar TSS[ld lu; rd ru] := loop_S_array[2*T_site-1]'[(); lu l u] * loop_S_array[2*T_site]'[(); u r ru] * loop_T_array[T_site][ld l r rd]
                    loop_TSS_array[T_site] = TSS
                end
            else 
                break
            end
            lastcost = cost
        end
    end

    """
    The last step of Loop-TNR: contracting four S-tensors into new TA, TB tensors.
    """
    function coares_grain(loop_S_array)
        @planar A[l u r d] := loop_S_array[4][r mr mu] * loop_S_array[5][mu ml u] * loop_S_array[8][l ml md] * loop_S_array[1][md mr d]
        @planar B[l u r d] := loop_S_array[2][u mu ml] * loop_S_array[3][ml md l] * loop_S_array[6][d md mr] * loop_S_array[7][mr mu r]
        return A, B
    end

    """
    Perform a single step of Loop-TNR.
    Dcut: cutoff
    N_sweep: maximal number of sweeps of the loop
    relative_descend: when the relative relative descend of error is smaller than the given one, quit the sweep
    absolute_error: when the absolute_error is smaller than the given one, quit the sweep
    
    return: next A, B
    """
    function loop(A, B, Dcut::Int, N_sweep::Int; relative_descend::Float64 = 0.02, absolute_error::Float64 = 1e-8)
        loop_T_array, loop_S_array, loop_TT_array, loop_SS_array, loop_TSS_array = loop_initialization(A, B, Dcut)
        sweep!(loop_T_array, loop_S_array, loop_TT_array, loop_SS_array, loop_TSS_array, N_sweep, relative_descend, absolute_error)
        return coares_grain(loop_S_array)
    end

end
