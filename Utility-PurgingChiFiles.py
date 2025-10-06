import os
import shutil

def purge_chi_output_folders(root_directory):
    """
    Recursively finds and deletes all folders named 'ChiOutput' within a given root directory.

    Args:
        root_directory (str): The absolute path to the directory to start searching from.
    """
    folders_deleted = 0
    # Walk through the directory tree top-down
    # We walk top-down so we can modify dirnames to prevent os.walk from descending
    # into a directory we are about to delete.
    for dirpath, dirnames, filenames in os.walk(root_directory):
        # Check if 'ChiOutput' is in the list of directories for the current path
        if 'ChiOutput' in dirnames:
            # Construct the full path to the target folder
            folder_to_delete = os.path.join(dirpath, 'ChiOutput')
            
            print(f"Found: {folder_to_delete}")
            
            try:
                # Use shutil.rmtree to recursively delete the folder and its contents
                shutil.rmtree(folder_to_delete)
                print(f"  -> DELETED successfully.")
                folders_deleted += 1
                # Optional: Remove 'ChiOutput' from dirnames to prevent os.walk
                # from trying to descend into the just-deleted directory.
                dirnames.remove('ChiOutput')
            except OSError as e:
                # Catch potential errors like permission denied
                print(f"  -> ERROR deleting {folder_to_delete}: {e}")

    if folders_deleted == 0:
        print("\nNo folders named 'ChiOutput' were found in the specified directory.")
    else:
        print(f"\nPurge complete. A total of {folders_deleted} 'ChiOutput' folder(s) were deleted.")

def main():
    """
    Main function to get user input and run the purge process.
    """
    print("--- ChiOutput Folder Purge Script ---")
    print("This script will permanently delete all folders named 'ChiOutput'")
    print("within the directory you specify and all of its subdirectories.\n")
    
    # Get the target directory from the user
    target_dir = input("Please enter the full path to the root directory to clean: ").strip()
    
    # Validate that the path exists and is a directory
    if not os.path.isdir(target_dir):
        print(f"\nError: The path '{target_dir}' is not a valid directory.")
        return
        
    # Final confirmation from the user because this is a destructive action
    print(f"\nWARNING: You are about to delete folders from: {target_dir}")
    confirm = input("Are you absolutely sure you want to proceed? (yes/no): ").strip().lower()
    
    if confirm == 'yes':
        print("\nStarting the purge process...")
        purge_chi_output_folders(target_dir)
    else:
        print("\nOperation cancelled by user.")

if __name__ == "__main__":
    main()
