import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = "https://hlgbxtguihuzcstzujnj.supabase.co";
const SUPABASE_ANON_KEY = "sb_publishable_LNI6coVXzUcLMipou5WQZw_Eb5Ftt04";

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

// Read the SQL schema file
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function setupSupabase() {
  console.log('🔧 Setting up Supabase tables directly...\n');
  
  try {
    // Read the SQL schema
    const schemaPath = join(__dirname, 'win-unpacked', 'resources', 'app_asar_extracted', 'server', 'supabase_schema.sql');
    const sqlContent = readFileSync(schemaPath, 'utf8');
    
    // Split into individual commands (simple approach)
    const commands = sqlContent
      .split(';')
      .map(cmd => cmd.trim())
      .filter(cmd => cmd.length > 0 && !cmd.startsWith('--'));
    
    console.log(`Found ${commands.length} SQL commands to execute\n`);
    
    // Execute each command
    for (let i = 0; i < commands.length; i++) {
      const command = commands[i];
      const cleanCommand = command.replace(/\s+/g, ' ').trim();
      
      if (cleanCommand.length < 10) continue; // Skip very short commands
      
      console.log(`Executing ${i + 1}/${commands.length}: ${cleanCommand.substring(0, 60)}...`);
      
      // Use rpc to execute raw SQL
      const { data, error } = await supabase.rpc('exec_sql', { sql: cleanCommand });
      
      if (error) {
        console.log(`   ⚠️  Error: ${error.message}`);
        // Continue anyway - some commands might fail but others work
      } else {
        console.log(`   ✅ Success`);
      }
    }
    
    console.log('\n🎉 Setup complete!');
    
  } catch (error) {
    console.error('❌ Setup failed:', error.message);
  }
}

setupSupabase();