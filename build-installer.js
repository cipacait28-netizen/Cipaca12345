#!/usr/bin/env node

import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

console.log('🚀 Building ProjectFlow Medical CRM Installer with Pre-configured Supabase...\n');

// Check if required files exist
const requiredFiles = [
  'win-unpacked/resources/app_asar_extracted/server/supabase_config.json',
  'win-unpacked/resources/app_asar_extracted/electron-main.cjs',
  'win-unpacked/resources/app_asar_extracted/server/server.js',
  'win-unpacked/resources/app_asar_extracted/dist/index.html'
];

console.log('📋 Checking required files...');
let allFilesExist = true;

for (const file of requiredFiles) {
  const filePath = path.join(__dirname, file);
  if (fs.existsSync(filePath)) {
    console.log(`✅ ${file}`);
  } else {
    console.log(`❌ ${file} - MISSING`);
    allFilesExist = false;
  }
}

if (!allFilesExist) {
  console.error('\n❌ Some required files are missing. Please ensure the project structure is correct.');
  process.exit(1);
}

// Verify Supabase configuration
console.log('\n🔐 Checking Supabase configuration...');
const supabaseConfigPath = path.join(__dirname, 'win-unpacked/resources/app_asar_extracted/server/supabase_config.json');
try {
  const config = JSON.parse(fs.readFileSync(supabaseConfigPath, 'utf8'));
  if (config.url && config.anonKey && config.enabled) {
    console.log('✅ Supabase configuration is valid and enabled');
    console.log(`   URL: ${config.url}`);
    console.log(`   Enabled: ${config.enabled}`);
  } else {
    console.log('⚠️  Supabase configuration may be incomplete');
  }
} catch (error) {
  console.error('❌ Error reading Supabase config:', error.message);
  process.exit(1);
}

// Create build directory if it doesn't exist
const buildDir = path.join(__dirname, 'build');
if (!fs.existsSync(buildDir)) {
  fs.mkdirSync(buildDir, { recursive: true });
  console.log('📁 Created build directory');
}

// Copy icon if it exists
const iconSource = path.join(__dirname, 'build/app.ico');
const iconDest = path.join(__dirname, 'app.ico');
if (fs.existsSync(iconSource)) {
  fs.copyFileSync(iconSource, iconDest);
  console.log('✅ Icon file copied');
} else {
  console.log('⚠️  Icon file not found at build/app.ico - using default');
}

console.log('\n🔨 Starting electron-builder...\n');

try {
  // Run electron-builder with the new configuration
  execSync('npx electron-builder --config electron-builder.json', {
    stdio: 'inherit',
    cwd: __dirname
  });
  
  console.log('\n✅ Installer built successfully!');
  console.log('📁 Check the dist_installer directory for your installer file.');
  
} catch (error) {
  console.error('\n❌ Build failed:', error.message);
  process.exit(1);
}