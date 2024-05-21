# Import the regular expression library
import re
# Import the CSV file handling library
import csv
# Import the operating system library, used for handling files and paths
import os
# Import the built-in sys library, used for handling operating system related information and commands
import sys


# Define the main function, args is the command line parameters, generally, we expect the first parameter to be the path to the CSV file
def cut_book3(csv_path, csv_name):
    # Define a regex pattern to match strings ending with an uppercase letter
    pattern_with_letter = r'[A-Z]$'
    # Get the CSV file path from the command line arguments
    csv_file = csv_path + csv_name
    # Open the CSV file in read-only mode, newline='' is to handle line breaks correctly in Windows
    with open(csv_file, 'r', newline='') as file:
        # Use the csv library's reader function to read the file content, reading each line into memory
        reader = csv.reader(file)
        # Convert all rows of data into a list for further processing
        data = list(reader)
        # Print the original data content for debugging and viewing the original data
        # print("\ndata",data)
        # Iterate over each row of data in the list
        for row in data:
            print("row", row)
            # If the data in the first column is 'Image_ID', skip this row and move to the next row
            if row[0] == 'Image_ID':
                continue
            # Use regex to match the data in the first column to see if there is data that matches the pattern
            id_match = re.search(pattern_with_letter, row[0].split('.')[0])
            print("id_match", id_match)
            # If data matching the pattern is found
            if id_match:
                # Match the data in the second column to see if there is a string matching the character following the string matched in the first column
                caption_match = re.search(
                    id_match.group(0) + r'(.*)' + chr(ord(id_match.group(0)) + 1), row[1])
                print("caption_match", caption_match)
                # If data matching the pattern is found
                if caption_match:
                    if row[2] == 'nan':
                        continue
                    else:
                        # Modify the data in the second column to be the matched string without the last character
                        row[1] = caption_match.group(0)[:-1]
                        # Modify the data in the third column to be the matched string without the last character, assuming the data in the third column exists in the form of the next character following the string matched in the first column
                        flag = re.search(id_match.group(0) + r'[\.．]+(.*)' + chr(ord(id_match.group(0)) + 1) + r'[\.．]', row[2])
                        # flag=re.search(id_match.group(0)+r'(.*)' + chr(ord(id_match.group(0))+1), row[2])
                        print("flag", flag)
                        if flag:
                            row[2] = re.search(id_match.group(0) + r'(.*)' + chr(ord(id_match.group(0)) + 1) + r'[\.．]', row[2]).group(0)[:-2]
                        else:
                            # row[2] = re.search(id_match.group(0)+r'(.*)' + chr(ord(id_match.group(0))+1)+r'[\.．]', row[2]).group(0)[:-2]
                            continue
                        print("row[2] ", row[2])
                else:
                    # If no data matching the pattern is found in the second and third columns, print the original data and the matching regex pattern for debugging and viewing the issue
                    print(f'row1:{row[1]},row2:{row[2]}')
                    print(id_match.group(0) + r'[\.．]+(.*)')
                    print(id_match.group(0) + r'[\.．]+(.*)')
                    # Find the form of the matching string in the second column and assign it to the data in the second column
                    row[1] = re.search(
                        id_match.group(0) + r'[\.．]+(.*)', row[1]).group(0)
                    # Find the form of the matching string in the third column and assign it to the data in the third column
                    if row[2] == 'nan':
                        continue
                    else:
                        row[2] = re.search(
                            id_match.group(0) + r'[\.．]+(.*)', row[2]).group(0)
        # Print the processed data content for viewing the processing results and debugging the code
        print(data)
        # Create a new CSV file, named by adding a '_result' suffix to the original CSV file name, to store the processed data
        writer = csv.writer(open(csv_path + csv_name.split('.')[0].split('_')[0] + '_result.csv', 'w', newline=''))
        # Write the processed data to the new CSV file, separating each row of data with a newline character
        writer.writerows(data)

def cut_book4(csv_path, csv_name):
    pattern_with_letter = r'[A-Z]\.'
    # Get the CSV file path from the command line arguments
    csv_file = csv_path + csv_name
    # Open the CSV file in read-only mode, newline='' is to handle line breaks correctly in Windows
    with open(csv_file, 'r', newline='') as file:
        # Use the csv library's reader function to read the file content, reading each line into memory
        reader = csv.reader(file)
        # Convert all rows of data into a list for further processing
        data = list(reader)
        # Print the original data content for debugging and viewing the original data
        # print("\ndata",data)
        # Iterate over each row of data in the list
        for row in data:
            print("row", row)
            # If the data in the first column is 'Image_ID', skip this row and move to the next row
            if row[0] == 'Image_ID':
                continue
            # Use regex to match the data in the first column to see if there is data that matches the pattern
            id_match = re.search(pattern_with_letter, row[0].split('.')[0])
            print("id_match", id_match)
            # If data matching the pattern is found
            if id_match:
                # Match the data in the second column to see if there is a string matching the character following the string matched in the first column
                caption_match = re.search(
                    id_match.group(0) + r'(.*)' + chr(ord(id_match.group(0)) + 1), row[1])
                print("caption_match", caption_match)
                # If data matching the pattern is found
                if caption_match:
                    if row[2] == 'nan':
                        continue
                    else:
                        # Modify the data in the second column to be the matched string without the last character
                        row[1] = caption_match.group(0)[:-1]
                        # Modify the data in the third column to be the matched string without the last character, assuming the data in the third column exists in the form of the next character following the string matched in the first column
                        flag = re.search(id_match.group(0) + r'[\.．]+(.*)' + chr(ord(id_match.group(0)) + 1) + r'[\.．]', row[2])
                        # flag=re.search(id_match.group(0)+r'(.*)' + chr(ord(id_match.group(0))+1), row[2])
                        print("flag", flag)
                        if flag:
                            row[2] = re.search(id_match.group(0) + r'(.*)' + chr(ord(id_match.group(0)) + 1) + r'[\.．]', row[2]).group(0)[:-2]
                        else:
                            # row[2] = re.search(id_match.group(0)+r'(.*)' + chr(ord(id_match.group(0))+1)+r'[\.．]', row[2]).group(0)[:-2]
                            continue
                        print("row[2] ", row[2])
                else:
                    # If no data matching the pattern is found in the second and third columns, print the original data and the matching regex pattern for debugging and viewing the issue
                    print(f'row1:{row[1]},row2:{row[2]}')
                    print(id_match.group(0) + r'[\.．]+(.*)')
                    print(id_match.group(0) + r'[\.．]+(.*)')
                    # Find the form of the matching string in the second column and assign it to the data in the second column
                    row[1] = re.search(
                        id_match.group(0) + r'[\.．]+(.*)', row[1]).group(0)
                    # Find the form of the matching string in the third column and assign it to the data in the third column
                    if row[2] == 'nan':
                        continue
                    else:
                        row[2] = re.search(
                            id_match.group(0) + r'[\.．]+(.*)', row[2]).group(0)
        # Print the processed data content for viewing the processing results and debugging the code
        print(data)
        # Create a new CSV file, named by adding a '_result' suffix to the original CSV file name, to store the processed data
        writer = csv.writer(open(csv_path + csv_name.split('.')[0].split('_')[0]+'_result.csv', 'w', newline=''))
        # Write the processed data to the new CSV file, separating each row of data with a newline character
        writer.writerows(data)

if __name__ == "__main__":
    book_path = './book4/'  #replace
    total_part_number = 1   #replace (the number of sub-PDFs)
    for i in range(total_part_number):
        path = book_path + 'caption/'
        csv_name = 'part' + str(i+1) + '_final.csv'
        cut_book4(path,csv_name)    #replace