# Module 2: Pondering Paths

**Overview:**  
The "Pondering Paths" module focuses on understanding the Linux filesystem, navigating directories, and managing file paths. This module emphasizes using relative and absolute paths effectively.

## Challenges

### Challenge 1: Absolute vs. Relative Paths
**Explanation:**  
Learn the difference between absolute and relative paths in the filesystem.

- **Absolute Path Example:**
  ```bash
  /home/hacker/documents/file.txt
  ```
- **Relative Path Example:**
  ```bash
  documents/file.txt
  ```

### Challenge 2: Navigating Directories
**Explanation:**  
Practice navigating through directories using commands like cd and ls.

- **Change Directory Example:**
  ```bash
  hacker@dojo:~$ cd /home/hacker/documents
  ```
- **List Contents:**
  ```bash
  hacker@dojo:~/documents$ ls
  ```

### Challenge 3: Creating Directories
**Explanation:**  
Learn how to create new directories with the mkdir command.

- **Create Directory Example:**
  ```bash
  hacker@dojo:~$ mkdir new_folder
  ```

### Challenge 4: Removing Files and Directories
**Explanation:**  
Understand how to delete files and directories using rm and rmdir.

- **Remove File Example:**
  ```bash
  hacker@dojo:~/documents$ rm file.txt
  ```
- **Remove Directory Example:**
  ```bash
  hacker@dojo:~$ rmdir new_folder
  ```

### Challenge 5: Copying Files
**Explanation:**  
Use the cp command to copy files from one location to another.

- **Copy File Example:**
  ```bash
  hacker@dojo:~$ cp original.txt /home/hacker/documents/
  ```

### Challenge 6: Moving Files
**Explanation:**  
Learn to move or rename files using the mv command.

- **Move File Example:**
  ```bash
  hacker@dojo:~$ mv file.txt /home/hacker/documents/
  ```

### Challenge 7: Finding Files
**Explanation:**  
Utilize the find command to locate files within the filesystem.

- **Find File Example:**
  ```bash
  hacker@dojo:~$ find /home/hacker -name "file.txt"
  ```

### Challenge 8: Viewing File Content
**Explanation:**  
Learn how to view the contents of files using commands like cat, less, and more.

- **View File Example:**
  ```bash
  hacker@dojo:~$ cat file.txt
  ```

### Challenge 9: Using Wildcards
**Explanation:**  
Understand how to use wildcards (*, ?) to match filenames in commands.

- **Wildcard Example:**
  ```bash
  hacker@dojo:~$ ls *.txt  # List all .txt files
  ```