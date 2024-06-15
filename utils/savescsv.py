import csv




# Sample dictionary
data = {
    "name": "John",
    "age": 30,
    "city": "New York"
}

def savedict(data, name):
    # File path to save the CSV
    file_path = "./patch_{}.csv".format(name)

    # Writing dictionary to CSV file
    with open(file_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=data.keys())
        
        # Write header
        writer.writeheader()
        
        # Write data
        writer.writerow(data)

    print("Dictionary saved in CSV format at:", file_path)
    
def savelist(data, name):
    # File path to save the CSV
    file_path = "./patch_{}.csv".format(name)
    # Writing list to CSV file
    with open(file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        # Write data
        writer.writerow(["Value"])  # Write header
        writer.writerows(zip(data))  # Write data

    print("List saved in CSV format at:", file_path)
        
