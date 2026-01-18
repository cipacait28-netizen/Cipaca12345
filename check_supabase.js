import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = "https://hlgbxtguihuzcstzujnj.supabase.co";
const SUPABASE_ANON_KEY = "sb_publishable_LNI6coVXzUcLMipou5WQZw_Eb5Ftt04";

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

async function checkData() {
  console.log('🔍 Checking Supabase database...\n');
  
  // Get all leads regardless of user_id
  const { data: allLeads, error: leadsError } = await supabase
    .from('leads')
    .select('*');
  
  if (leadsError) {
    console.log('❌ Error fetching leads:', leadsError);
  } else {
    console.log('📊 Total leads in database:', allLeads?.length || 0);
    if (allLeads && allLeads.length > 0) {
      console.log('\n📋 Sample leads:');
      allLeads.slice(0, 5).forEach(lead => {
        console.log(`  - ID: ${lead.id} | User: ${lead.user_id} | Doctor: ${lead.doctor_name || 'N/A'}`);
      });
    }
  }

  // Check for leads with shared_user_001
  const { data: sharedLeads, error: sharedError } = await supabase
    .from('leads')
    .select('*')
    .eq('user_id', 'shared_user_001');
  
  console.log('\n✅ Leads with shared_user_001:', sharedLeads?.length || 0);

  // Check for leads with old user IDs
  const { data: oldLeads, error: oldError } = await supabase
    .from('leads')
    .select('*')
    .like('user_id', 'user_%');
  
  console.log('⚠️ Leads with old user IDs:', oldLeads?.length || 0);
  if (oldLeads && oldLeads.length > 0) {
    const uniqueOldUsers = [...new Set(oldLeads.map(l => l.user_id))];
    console.log('   Old user IDs found:', uniqueOldUsers);
  }

  // Check stages too
  const { data: allStages, error: stagesError } = await supabase
    .from('stages')
    .select('*');
  
  console.log('\n📊 Total stages in database:', allStages?.length || 0);
  
  const { data: sharedStages, error: sharedStagesError } = await supabase
    .from('stages')
    .select('*')
    .eq('user_id', 'shared_user_001');
  
  console.log('✅ Stages with shared_user_001:', sharedStages?.length || 0);
}

checkData().catch(console.error);