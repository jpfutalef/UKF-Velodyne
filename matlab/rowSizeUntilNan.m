function n = rowSizeUntilNan(row)
a = size(row,2);

for i = 1:a
    if isnan(row(1, i))
        break;
    end
end

if i == a
    n = i;
else
    n = i-1; 
end
end