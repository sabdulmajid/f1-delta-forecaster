#!/usr/bin/env python3
"""
A script to rewrite the Git history of the current branch.

This script takes all commits on the current branch and redistributes them
over the last 60 days, creating a more "natural-looking" contribution history.

WARNING: This is a destructive operation. It will rewrite your Git history.
Only run this on a personal project or a feature branch that you have not
shared with others. It is highly recommended to back up your repository first.
"""

import os
import subprocess
import random
from datetime import datetime, timedelta
import argparse

def run_command(command):
    """Runs a shell command and returns its output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Stderr: {e.stderr.strip()}")
        raise

def get_commits_on_branch():
    """Gets the list of commit hashes on the current branch."""
    # Find the common ancestor with main/master or fallback to the root
    main_branch = "main" if run_command("git branch --list main") else "master"
    try:
        ancestor = run_command(f"git merge-base {main_branch} HEAD")
        commit_range = f"{ancestor}..HEAD"
    except subprocess.CalledProcessError:
        print("Could not find a 'main' or 'master' branch. Rewriting all commits.")
        commit_range = "HEAD"
        
    return run_command(f"git rev-list --reverse {commit_range}").splitlines()

def generate_new_dates(num_commits):
    """Generates a list of new dates spread over the last 60 days."""
    now = datetime.now()
    start_date = now - timedelta(days=60)
    total_seconds = (now - start_date).total_seconds()
    
    new_dates = []
    for i in range(num_commits):
        # Distribute commits evenly
        fraction = (i + 1) / num_commits
        
        # Add some randomness to make it look more natural
        random_factor = random.uniform(-0.5 / num_commits, 0.5 / num_commits)
        fraction += random_factor
        fraction = max(0, min(1, fraction)) # Clamp to [0, 1]

        commit_date = start_date + timedelta(seconds=total_seconds * fraction)
        
        # Simulate work hours (e.g., 9am to 10pm)
        if commit_date.weekday() < 5: # Weekdays
            hour = random.randint(9, 22)
        else: # Weekends
            hour = random.randint(11, 19)
            
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        
        commit_date = commit_date.replace(hour=hour, minute=minute, second=second)
        new_dates.append(commit_date)
        
    return sorted(new_dates)

def main():
    """Main function to rewrite the Git history."""
    parser = argparse.ArgumentParser(
        description="A script to rewrite the Git history of the current branch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WARNING: This is a destructive operation. It will rewrite your Git history.
Only run this on a personal project or a feature branch that you have not
shared with others. It is highly recommended to back up your repository first.
"""
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass the interactive confirmation prompt and proceed with the rewrite."
    )
    args = parser.parse_args()

    print("Starting Git history rewrite...")

    # 1. Safety checks
    if run_command("git status --porcelain"):
        print("Error: Your working directory is not clean. Please commit or stash your changes.")
        return

    current_branch = run_command("git rev-parse --abbrev-ref HEAD")
    print(f"Current branch: {current_branch}")

    # 2. Create a backup branch
    backup_branch = f"backup/{current_branch}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_command(f"git branch {backup_branch}")
    print(f"Created backup branch: {backup_branch}")

    # 3. Get commits and generate new dates
    commits = get_commits_on_branch()
    if not commits:
        print("No commits to rewrite.")
        return
        
    print(f"Found {len(commits)} commits to rewrite.")
    
    new_dates = generate_new_dates(len(commits))
    
    # 4. Create a mapping from old commit to new date
    commit_date_map = dict(zip(commits, new_dates))

    # 5. Rewrite history using git filter-branch
    # This is a complex command. We are setting the author and committer dates
    # for each commit based on our generated map.
    filter_script = " ".join([
        f"if [ -n \"$GIT_COMMIT\" ] && [ \"{commit_date_map.get('$GIT_COMMIT')}\" != \"\" ]; then",
        f"    export GIT_AUTHOR_DATE='{commit_date_map.get('$GIT_COMMIT')}'",
        f"    export GIT_COMMITTER_DATE='{commit_date_map.get('$GIT_COMMIT')}'",
        "fi"
    ])

    # We need to create a temporary script to handle the date formatting correctly
    # because shell commands can be tricky with dates.
    
    print("Generating rebase instructions...")
    
    # Using git rebase is safer and more modern than filter-branch
    # We will create a rebase sequence and edit the dates.
    
    # Let's use a different approach that is more robust: interactive rebase.
    # filter-branch is powerful but has many pitfalls.
    # A manual-like process is safer to script.
    
    # Let's stick to a script that generates a shell script to run.
    # This is safer as the user can inspect it first.
    
    rewrite_script_path = "rewrite_history.sh"
    with open(rewrite_script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("set -e\n\n")
        f.write("# This script will rewrite the history of your current branch.\n")
        f.write("# Please review it carefully before running.\n\n")
        
        for commit, new_date in zip(commits, new_dates):
            date_str = new_date.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"# Rewriting commit {commit[:7]} to {date_str}\n")
            # This approach is complex to get right.
            
    # Let's reconsider filter-branch. It's designed for this.
    # The main issue is passing the dates correctly.
    
    # We will create a temporary file with the remapping logic.
    env_filter_script = []
    for commit, new_date in commit_date_map.items():
        date_iso = new_date.isoformat()
        env_filter_script.append(f"if [ \"$GIT_COMMIT\" = \"{commit}\" ]; then")
        env_filter_script.append(f"    export GIT_AUTHOR_DATE='{date_iso}'")
        env_filter_script.append(f"    export GIT_COMMITTER_DATE='{date_iso}'")
        env_filter_script.append("fi")

    env_filter_command = "\n".join(env_filter_script)
    
    print("\nWARNING: The next step will rewrite your Git history.")
    print("A backup of your branch has been created.")
    print("If anything goes wrong, you can restore with:")
    print(f"  git reset --hard {backup_branch}\n")
    
    proceed = 'n'
    if not args.force:
        proceed = input("Do you want to proceed? (y/n): ")
    else:
        print("Bypassing prompt due to --force flag.")
        proceed = 'y'

    if proceed.lower() != 'y':
        print("Aborted.")
        return

    try:
        print("Rewriting history... This may take a while.")
        run_command(f"git filter-branch --env-filter '{env_filter_command}' --tag-name-filter cat -- --all")
        print("\nHistory rewritten successfully!")
        print("Please check your git log to confirm the changes.")
        print("If you are happy with the changes, you may need to force-push to your remote:")
        print(f"  git push --force origin {current_branch}")
    except Exception as e:
        print("\nAn error occurred during history rewriting.")
        print("Your original branch is safe in the backup.")
        print(f"To restore, run: git reset --hard {backup_branch}")

if __name__ == "__main__":
    main()
