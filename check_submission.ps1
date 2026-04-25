# OpenEnv Hackathon 2026 - Submission Validation
# PR Pilot - Code Review Environment

Write-Host ""
Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   OpenEnv Hackathon 2026 - Submission Validation" -ForegroundColor Cyan
Write-Host "   PR Pilot - Multi-Agent Code Review Environment" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$checked = 0
$passed = 0
$failed = 0

function Check-Item {
    param(
        [string]$Name,
        [bool]$Condition,
        [string]$Level = "Required"
    )
    
    $script:checked++
    Write-Host "  $Name : " -NoNewline
    
    if ($Condition) {
        Write-Host "✓" -ForegroundColor Green
        $script:passed++
    } else {
        if ($Level -eq "Optional") {
            Write-Host "⚠ Missing (optional)" -ForegroundColor Yellow
        } else {
            Write-Host "✗ MISSING" -ForegroundColor Red
            $script:failed++
        }
    }
}

# ============================================================
# JUDGING CRITERIA
# ============================================================

Write-Host "📊 JUDGING CRITERIA (100 points total)" -ForegroundColor Yellow
Write-Host ""

Write-Host "  [40 pts] Environment Innovation" -ForegroundColor White
Check-Item "    Multi-agent architecture" (Test-Path "code_review_env\env.py")
Check-Item "    Debate simulation"  ((Select-String -Path "code_review_env\state.py" -Pattern "debate" -Quiet 2>$null) -eq $true)
Check-Item "    Reward shaping" (Test-Path "code_review_env\reward.py")
Write-Host ""

Write-Host "  [30 pts] Storytelling & Presentation" -ForegroundColor White
$hasStory = (Test-Path "HACKATHON_BLOG.md") -or (Test-Path "HACKATHON_VIDEO_SCRIPT.md") -or (Test-Path "HACKATHON_SLIDES.md")
Check-Item "    Blog/Video/Slides materials ready" $hasStory
Check-Item "    README has storytelling section" ((Select-String -Path "README.md" -Pattern "Storytelling" -Quiet 2>$null) -eq $true)
Write-Host "    ⚠ Remember to PUBLISH and add URL to README!" -ForegroundColor Yellow
Write-Host ""

Write-Host "  [20 pts] Training Evidence" -ForegroundColor White
Check-Item "    training_summary.txt" (Test-Path "results\training_summary.txt")
Check-Item "    training_plot.png" (Test-Path "results\training_plot.png")
Check-Item "    baseline_comparison.png" (Test-Path "results\baseline_comparison.png")
Write-Host ""

Write-Host "  [10 pts] Training Pipeline" -ForegroundColor White
Check-Item "    TRL training notebook" (Test-Path "training_trl_colab.ipynb")
Check-Item "    Policy gradient implementation" ((Select-String -Path "training_trl_colab.ipynb" -Pattern "Adam|policy" -Quiet 2>$null) -eq $true)
Write-Host ""

# ============================================================
# MINIMUM REQUIREMENTS
# ============================================================

Write-Host "✅ MINIMUM REQUIREMENTS" -ForegroundColor Yellow
Write-Host ""

Check-Item "OpenEnv framework (openenv.yaml)" (Test-Path "openenv.yaml")
Check-Item "Training script with TRL" (Test-Path "training_trl_colab.ipynb")  
Check-Item "Training evidence files" (Test-Path "results\training_summary.txt")
Check-Item "README.md" (Test-Path "README.md")
Check-Item "Dockerfile for deployment" (Test-Path "Dockerfile")
Write-Host ""

# ============================================================
# DEPLOYMENT
# ============================================================

Write-Host "🚀 DEPLOYMENT" -ForegroundColor Yellow
Write-Host ""

$hasHFLink = (Select-String -Path "README.md" -Pattern "huggingface.co/spaces" -Quiet 2>$null) -eq $true
Check-Item "HuggingFace Space link in README" $hasHFLink "Optional"
Check-Item "Colab badge in README" ((Select-String -Path "README.md" -Pattern "colab-badge" -Quiet 2>$null) -eq $true)
Check-Item "Deployment guide" (Test-Path "DEPLOY_TO_HUGGINGFACE.md") "Optional"
Write-Host ""

# ============================================================
# DOCUMENTATION
# ============================================================

Write-Host "📚 DOCUMENTATION" -ForegroundColor Yellow
Write-Host ""

Check-Item "Training guide (COLAB_INSTRUCTIONS.md)" (Test-Path "COLAB_INSTRUCTIONS.md") "Optional"
Check-Item "Quick start guide" (Test-Path "QUICK_START_COLAB.md") "Optional"
Check-Item "Results README" (Test-Path "results\README.md") "Optional"
Write-Host ""

# ============================================================
# SUMMARY
# ============================================================

Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   SUMMARY: $passed passed, $failed critical missing" -ForegroundColor $(if ($failed -eq 0) { "Green" } else { "Yellow" })
Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

if ($failed -eq 0) {
    Write-Host "✅ All critical items present! Ready for submission." -ForegroundColor Green
} else {
    Write-Host "⚠ $failed critical items missing - see ✗ marks above" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "📋 NEXT STEPS (Priority Order):" -ForegroundColor Yellow
Write-Host ""

if (-not (Test-Path "results\training_plot.png")) {
    Write-Host "1. ❌ Download training plots from Colab" -ForegroundColor Red
    Write-Host "   → Run: .\download_and_commit_results.ps1" -ForegroundColor Gray
} else {
    Write-Host "1. ✅ Training plots downloaded" -ForegroundColor Green
}

if (-not $hasStory) {
    Write-Host "2. ❌ Publish storytelling content (30 pts!)" -ForegroundColor Red
    Write-Host "   Choose: HACKATHON_BLOG.md OR VIDEO_SCRIPT.md OR SLIDES.md" -ForegroundColor Gray
} else {
    Write-Host "2. ⚠ Storytelling materials ready - PUBLISH and add URL!" -ForegroundColor Yellow
}

if (-not $hasHFLink) {
    Write-Host "3. ⚠ Deploy to HuggingFace Space" -ForegroundColor Yellow
    Write-Host "   → See: DEPLOY_TO_HUGGINGFACE.md" -ForegroundColor Gray
} else {
    Write-Host "3. ✅ HuggingFace Space configured" -ForegroundColor Green
}

Write-Host ""
Write-Host "🎯 Final Check: Test HF Space URL manually before submission!" -ForegroundColor Yellow
Write-Host ""
