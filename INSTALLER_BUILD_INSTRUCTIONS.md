# ProjectFlow Medical CRM - Installer Build Instructions

This guide will help you build a new installer that includes pre-configured Supabase settings.

## Prerequisites

- Node.js (v18 or higher)
- npm or yarn
- Windows operating system (for building Windows installer)

## Current Setup Status

Your current Supabase configuration is already set up and working:
- **Supabase URL**: `https://hlgbxtguihuzcstzujnj.supabase.co`
- **Status**: Enabled
- **API Key**: Configured in `supabase_config.json`

## Building the Installer

### Method 1: Using the Build Script (Recommended)

1. Open a terminal in the project root directory
2. Run the build script:
   ```bash
   node build-installer.js
   ```

This will:
- ✅ Verify all required files exist
- ✅ Check Supabase configuration
- ✅ Build the installer with pre-configured settings
- ✅ Output the installer to `dist_installer/`

### Method 2: Using electron-builder directly

1. Install dependencies (if not already done):
   ```bash
   npm install
   ```

2. Build the installer:
   ```bash
   npx electron-builder --config electron-builder.json
   ```

## What Gets Included

The new installer will include:

1. **Pre-configured Supabase settings** - No setup required after installation
2. **All frontend assets** - The web interface
3. **Backend server** - Express server with all sync capabilities
4. **Database schema** - Ready to connect to your Supabase project

## Installation Process for End Users

When users run the new installer:

1. They choose installation directory
2. Software installs with all Supabase settings pre-configured
3. Application launches and connects immediately to your Supabase database
4. **No additional setup required** - works out of the box

## Live Data Sync

✅ **Yes, live data sync works!** All devices using this installer will:
- Share the same Supabase database
- See real-time updates from other users
- Sync data instantly across all installations

## Files Created

- `electron-builder.json` - New build configuration
- `build-installer.js` - Build automation script
- `dist_installer/` - Output directory for installer files

## Troubleshooting

If the build fails:
1. Ensure all required files exist in the correct locations
2. Check that `supabase_config.json` is valid JSON
3. Verify Node.js version is 18 or higher
4. Run `npm install` to ensure dependencies are installed

## Next Steps

1. Run the build script to create your installer
2. Test the installer on a different machine
3. Verify live data sync works between devices
4. Distribute the installer to users