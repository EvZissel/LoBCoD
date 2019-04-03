function [ a ] = hard_threshold( a,t )
    for i=1:size(a,1)
        for j=1:size(a,2)
            a{i,j}(abs(a{i,j})<t)=0;        
        end
    end
end

