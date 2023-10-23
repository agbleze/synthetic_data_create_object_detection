


from argparse import ArgumentParser



#%%%
import os
if __name__ == '__main__':
    arg = ArgumentParser(description="change env file value from command line")
    arg.add_argument("--num_epochs", "-n",
                     required=True,
                     type=int,
                     dest="num_epochs",
                     )
    
    
    args_collected = arg.parse_args()
    
    if ".env" in os.listdir():
        print(f"env file found. will write changes num_epochs={args_collected.num_epochs}")
        
        with open(".env", "a+") as env_file:
            env_file.write(f"num_epochs={args_collected.num_epochs}")
        print(f"successfully written to env file")
    else:
        print("env file not found")
        
        
        
        
        
        
# # Importing FileInput from fileinput module 
# from fileinput import FileInput 

# # Creating a function to 
# # replace the text 
# #def replacetext(search_text, replace_text): 

# 	# Opening file using FileInput 
# 	with FileInput("SampleFile.txt", inplace=True, 
# 				backup='.bak') as f: 

# 		# Iterating over every and changing 
# 		# the search_text with replace_text 
# 		# using the replace function 
# 		for line in f: 
# 			print(line.replace(search_text, 
# 							replace_text), end='') 

# 	# Return "Text replaced" string 
# 	return "Text replaced"


# # Creating a variable and storing 
# # the text that we want to search 
# search_text = "dummy"

# # Creating a variable and storing 
# # the text that we want to update 
# replace_text = "replaced"

# # Calling the replacetext function 
# # and printing the returned statement 
# print(replacetext(search_text, replace_text)) 
