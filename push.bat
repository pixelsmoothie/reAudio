@echo off
setlocal enabledelayedexpansion
title QUED Git Auto-Push
color 0A

echo.
echo  ============================================
echo   GIT AUTO-PUSH by Niraj
echo  ============================================
echo.

:: ── CHECK IF GIT IS INSTALLED ──────────────────
where git >nul 2>&1
if errorlevel 1 (
    color 0C
    echo  [ERROR] Git is not installed or not in PATH.
    pause
    exit /b 1
)

:: ── CHECK IF INSIDE A GIT REPO ─────────────────
git rev-parse --git-dir >nul 2>&1
if errorlevel 1 (
    echo  [INFO] No git repo found. Initializing...
    echo.
    git init
    if errorlevel 1 ( echo  [ERROR] git init failed. & pause & exit /b 1 )
    echo.
    echo  [OK] Git repo initialized.
    echo.

    :: Ask for repo name
    set /p REPO_NAME= Enter GitHub repo name: 
    if "!REPO_NAME!"=="" (
        echo  [ERROR] Repo name cannot be empty.
        pause
        exit /b 1
    )

    :: Ask for visibility
    echo.
    echo  Visibility:
    echo    [1] Public
    echo    [2] Private
    set /p VIS_CHOICE= Choose (1 or 2): 
    if "!VIS_CHOICE!"=="1" ( set VISIBILITY=public ) else ( set VISIBILITY=private )

    echo.
    echo  Creating GitHub repo: !REPO_NAME! ^(!VISIBILITY!^)...
    echo.

    :: Create repo via GitHub CLI
    where gh >nul 2>&1
    if errorlevel 1 (
        color 0E
        echo  [WARN] GitHub CLI ^(gh^) not found.
        echo  Install it from: https://cli.github.com
        echo  Skipping remote creation. You will need to add it manually.
        echo.
        set SKIP_REMOTE=1
    ) else (
        gh repo create !REPO_NAME! --!VISIBILITY! --source=. --remote=origin --push
        if errorlevel 1 (
            echo  [ERROR] GitHub repo creation failed.
            pause
            exit /b 1
        )
        echo.
        echo  [OK] GitHub repo created and pushed.
        goto :DONE
    )

    if "!SKIP_REMOTE!"=="1" (
        echo  [INFO] Skipped remote. Add it manually with:
        echo        git remote add origin https://github.com/YOUR_USERNAME/!REPO_NAME!.git
        echo.
    )
) 

:: ── ALREADY A REPO — JUST COMMIT AND PUSH ──────
echo  [INFO] Git repo detected.
echo.

:: Show current status
echo  ── git status ──────────────────────────────
git status
echo  ────────────────────────────────────────────
echo.

:: Check if there's anything to commit
git diff --quiet && git diff --cached --quiet
if errorlevel 0 (
    git status --porcelain | findstr /r "." >nul 2>&1
    if errorlevel 1 (
        echo  [INFO] Nothing to commit. Working tree is clean.
        echo.
        goto :PUSH_ONLY
    )
)

:: Ask for commit message
set /p COMMIT_MSG= Enter commit message (leave blank for "wip"): 
if "!COMMIT_MSG!"=="" set COMMIT_MSG=wip

echo.
echo  ── git add . ───────────────────────────────
git add .
if errorlevel 1 ( echo  [ERROR] git add failed. & pause & exit /b 1 )
echo  [OK] All files staged.
echo.

echo  ── git commit ──────────────────────────────
git commit -m "!COMMIT_MSG!"
if errorlevel 1 ( echo  [ERROR] git commit failed. & pause & exit /b 1 )
echo.

:PUSH_ONLY
:: Check if remote exists
git remote get-url origin >nul 2>&1
if errorlevel 1 (
    color 0E
    echo  [WARN] No remote origin found.
    set /p REMOTE_URL= Enter remote URL to add: 
    if "!REMOTE_URL!"=="" (
        echo  [ERROR] No remote URL provided. Skipping push.
        goto :DONE
    )
    git remote add origin !REMOTE_URL!
    echo  [OK] Remote added.
    echo.
)

echo  ── git push ────────────────────────────────
:: Try pushing to current branch, fallback to main/master
for /f %%i in ('git branch --show-current 2^>nul') do set CURRENT_BRANCH=%%i
if "!CURRENT_BRANCH!"=="" set CURRENT_BRANCH=main

git push origin !CURRENT_BRANCH!
if errorlevel 1 (
    echo.
    echo  [INFO] Push failed. Trying with --set-upstream...
    git push --set-upstream origin !CURRENT_BRANCH!
    if errorlevel 1 (
        echo  [ERROR] Push failed. Check your remote and credentials.
        pause
        exit /b 1
    )
)
echo.

:DONE
echo  ============================================
echo   DONE. Code is safe on GitHub.
echo  ============================================
echo.
pause
