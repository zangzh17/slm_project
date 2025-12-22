function I_db = pow2dbn(I,db_min)
if nargin<2
    db_min = -40;
end
I = I/max(I(:));
I(I<10^(db_min/10))=10^(db_min/10);
I_db = pow2db(I); 
