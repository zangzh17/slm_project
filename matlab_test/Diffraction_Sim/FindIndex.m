function order_ind_N = FindIndex(M,N,order_ind_M)
ord_M = FourierOrder(M);
ord_N = FourierOrder(N);
order_ind_N = arrayfun(@(x) find(x==ord_N),ord_M(order_ind_M));
end