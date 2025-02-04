def find_unique_elements(*lists):
    unique_elements = set()
    all_elements = set()
    
    for lst in lists:
        for item in lst:
            if item in all_elements:
                unique_elements.discard(item)
            else:
                all_elements.add(item)
                unique_elements.add(item)
    
    return unique_elements


list1 = [1, 2, 3, 4]
list2 = [3, 4, 5, 6]
list3 = [6, 7, 8, 9]

unique = find_unique_elements(list1, list2, list3)
print("Унікальні елементи:", unique)