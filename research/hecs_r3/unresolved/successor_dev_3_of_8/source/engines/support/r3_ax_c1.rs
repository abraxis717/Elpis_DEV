//! Isolated successor candidate. NOT imported by historical release engine.
use hecs_r2::model::{row,DIM,LATENT};
use super::r3_training_core::{self,Window,Objective,Work};
fn dp(u:f64)->f64{0.5+u+1.5*u*u}
// Detached teacher reference: train-only true latent at t=3 is input to fourth teacher transition.
// Reference depends on W numerically, but has NO derivative in this objective by construction.
fn reference(w:&[f64],width:usize,win:&Window)->[f64;LATENT]{r3_training_core::forward(w,width,win.z[3],win.actions[3])}
fn c1_fixed(w:&[f64],width:usize,win:&Window,reference:[f64;LATENT],variance:f64)->Result<(f64,Vec<f64>,u64),String>{
 if width==0||w.len()!=DIM*width||win.z.len()<5||win.actions.len()<4||!variance.is_finite()||variance<=0.||!w.iter().all(|x|x.is_finite())||!win.actions.iter().take(4).all(|x|x.is_finite())||!win.z.iter().take(5).flatten().all(|x|x.is_finite())||!reference.iter().all(|x|x.is_finite()){return Err("INVALID_C1_INPUT".into())}
 let mut inputs=Vec::new();let mut state=win.z[0];
 for t in 0..4{inputs.push(state);state=r3_training_core::forward(w,width,state,win.actions[t]);if !state.iter().all(|v|v.is_finite())||state.iter().any(|v|v.abs()>1e3){return Err("C1_NUMERICAL_ESCAPE".into())}}
 let mut grad=vec![0.;w.len()];let mut adj=[0.;LATENT];let mut loss=0.;
 for k in 0..LATENT{let e=state[k]-reference[k];loss+=0.5*e*e/(LATENT as f64*variance);adj[k]=e/(LATENT as f64*variance)}
 for t in (0..4).rev(){let mut next=[0.;LATENT];for k in 0..LATENT{let x=row(&inputs[t],win.actions[t],k);for i in 0..width{let mut u=0.;for a in 0..DIM{u+=x[a]*w[a*width+i]};let v=adj[k]*dp(u);for a in 0..DIM{grad[a*width+i]+=v*x[a]};for a in 0..LATENT{next[a]+=v*w[a*width+i]}}}adj=next;}
 if !loss.is_finite()||!grad.iter().all(|v|v.is_finite()){return Err("NONFINITE_C1_LOSS_GRADIENT".into())}
 let rows=4*LATENT*width;let extra=DIM as u64*(2*rows) as u64+LATENT as u64*rows as u64;
 Ok((loss,grad,extra))
}
pub fn successor_step(w:&mut [f64],width:usize,windows:&[Window],variance:f64,base_lambda:f64,lambda4:f64,rate:f64,clip:f64)->Result<(f64,Work,u64),String>{
 if windows.is_empty()||!lambda4.is_finite()||lambda4<0.||!w.iter().all(|x|x.is_finite()){return Err("INVALID_SUCCESSOR_PARAMS".into())}
 // Exactly 4-step C1 windows, frozen reference is computed once before any update.
 if windows.iter().any(|x|x.z.len()!=5||x.actions.len()!=4){return Err("C1_REQUIRES_H4_WINDOW".into())}
 let (base,mut grad,work)=r3_training_core::objective_gradient(w,width,windows,4,variance,base_lambda,Objective::Rollout)?;
 let mut extra=0u64;let mut c1=0.;
 for win in windows{let detached=reference(w,width,win);let (loss,g,cost)=c1_fixed(w,width,win,detached,variance)?;c1+=loss/(windows.len() as f64);extra=extra.checked_add(cost).ok_or("LEDGER_OVERFLOW")?;for i in 0..grad.len(){grad[i]+=lambda4*g[i]/windows.len() as f64;}}
 let combined=base+lambda4*c1;if !combined.is_finite()||!grad.iter().all(|v|v.is_finite()){return Err("NONFINITE_COMBINED".into())}
 let mut candidate=w.to_vec();r3_training_core::update(&mut candidate,&grad,rate,clip)?;
 if !candidate.iter().all(|v|v.is_finite()){return Err("NONFINITE_UPDATED_CANDIDATE".into())}
 w.copy_from_slice(&candidate);Ok((combined,work,extra))
}
pub fn loss_fixed_reference(w:&[f64],width:usize,win:&Window,r:[f64;LATENT],var:f64)->Result<(f64,Vec<f64>),String>{let (l,g,_)=c1_fixed(w,width,win,r,var)?;Ok((l,g))}
pub fn detached_reference(w:&[f64],width:usize,win:&Window)->[f64;LATENT]{reference(w,width,win)}
