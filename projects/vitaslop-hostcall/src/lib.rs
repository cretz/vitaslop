//! `#[hostcall]`: write a Vita host handler as a normal typed Rust function and
//! let this macro generate the AAPCS-VFP argument marshalling and the return
//! write. The behavior stays entirely hand-written; only the boring register and
//! memory shuffle is generated, per the design in `vitaslop-runtime`'s README.
//!
//! # What you write
//! ```ignore
//! #[hostcall]
//! fn sce_kernel_alloc_mem_block(st: &mut VitaState, _name: Ptr, _ty: u32, size: u32, _opt: Ptr) -> i32 {
//!     st.alloc_memblock(size, 256 * 1024)
//! }
//! ```
//!
//! # What is generated
//! A `fn sce_kernel_alloc_mem_block(ctx: &mut GuestCtx, st: &mut VitaState)` that
//! reads each value parameter from the right register file - integer/pointer args
//! from the core registers (r0..r3 then the stack), float args from the VFP
//! registers (s0.. / d0..) because the Vita is hardfloat - runs your body, and
//! writes the typed return to r0 (or s0/d0 for a float return).
//!
//! # Parameter and return types
//! - `&mut VitaState` / `&VitaState`: the per-run host state, threaded through.
//! - `&mut GuestCtx` / `&GuestCtx`: the raw call context, for handlers that read
//!   or write guest memory directly (out-params, structs, strings).
//! - value args and returns: `u32`, `i32`, `bool`, `Ptr` (int class), `f32`,
//!   `f64` (float class). `()` return writes nothing.
//! Value parameters are consumed left to right in the declared order.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, FnArg, Ident, ItemFn, Pat, ReturnType, Type};

#[proc_macro_attribute]
pub fn hostcall(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    match expand(func) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// A value parameter or return type this macro knows how to marshal.
enum ValueKind {
    U32,
    I32,
    Bool,
    Ptr,
    F32,
    F64,
}

impl ValueKind {
    /// Classify a value type by its trailing path segment, e.g. `u32`, `Ptr`.
    fn from_type(ty: &Type) -> Option<ValueKind> {
        let name = type_last_ident(ty)?;
        Some(match name.to_string().as_str() {
            "u32" => ValueKind::U32,
            "i32" => ValueKind::I32,
            "bool" => ValueKind::Bool,
            "Ptr" => ValueKind::Ptr,
            "f32" => ValueKind::F32,
            "f64" => ValueKind::F64,
            _ => return None,
        })
    }
}

/// How each function parameter is supplied to the handler.
enum Param {
    /// `&mut VitaState` - the host state, bound to the wrapper's `__st`.
    State(Ident),
    /// `&mut GuestCtx` - the raw context, bound to the wrapper's `__ctx`.
    Ctx(Ident),
    /// A marshalled value argument: bind `ident` by reading the next arg slot.
    Value(Ident, ValueKind),
}

fn expand(func: ItemFn) -> syn::Result<proc_macro2::TokenStream> {
    let vis = &func.vis;
    let name = &func.sig.ident;
    let body = &func.block;

    if let Some(recv) = func.sig.receiver() {
        return Err(syn::Error::new_spanned(recv, "#[hostcall] does not take self"));
    }

    let mut params = Vec::new();
    for arg in &func.sig.inputs {
        let FnArg::Typed(pt) = arg else {
            return Err(syn::Error::new_spanned(arg, "#[hostcall] does not take self"));
        };
        let Pat::Ident(pat_ident) = pt.pat.as_ref() else {
            return Err(syn::Error::new_spanned(&pt.pat, "#[hostcall] arguments must be simple names"));
        };
        let ident = pat_ident.ident.clone();
        params.push(classify_param(ident, &pt.ty)?);
    }

    // The marshalling reads value args in order; the reborrows bind the user's
    // state/ctx names for the body; `__ctx` / `__st` are the wrapper's params.
    let mut marshal = Vec::new();
    let mut rebind = Vec::new();
    let mut uses_ctx = false;
    let mut uses_st = false;
    for p in &params {
        match p {
            Param::State(id) => {
                uses_st = true;
                rebind.push(quote! { let #id = &mut *__st; });
            }
            Param::Ctx(id) => {
                uses_ctx = true;
                rebind.push(quote! { let #id = &mut *__ctx; });
            }
            Param::Value(id, kind) => {
                uses_ctx = true;
                let read = read_expr(kind);
                marshal.push(quote! { let #id = #read; });
            }
        }
    }

    // A value return writes through `__ctx` (ret/ret_f32/ret_f64), so a handler
    // that only returns a value - no ctx or value args - still needs the context
    // bound with its real name.
    let uses_ctx = uses_ctx || matches!(func.sig.output, ReturnType::Type(..));
    let ctx_param = if uses_ctx { format_ident!("__ctx") } else { format_ident!("_ctx") };
    let st_param = if uses_st { format_ident!("__st") } else { format_ident!("_st") };

    // The body runs with the user's state/ctx bindings in a scope, so their
    // reborrows drop before the return write reclaims `__ctx`.
    let inner = quote! {
        {
            #( #rebind )*
            #body
        }
    };

    let ret = match &func.sig.output {
        ReturnType::Default => quote! { #inner; },
        ReturnType::Type(_, ty) => {
            let kind = ValueKind::from_type(ty).ok_or_else(|| {
                syn::Error::new_spanned(ty, "#[hostcall] return type must be u32, i32, bool, Ptr, f32, or f64")
            })?;
            let write = write_expr(&kind);
            quote! {
                let __ret = #inner;
                #write
            }
        }
    };

    Ok(quote! {
        #vis fn #name(
            #ctx_param: &mut ::vitaslop_runtime::host::GuestCtx,
            #st_param: &mut ::vitaslop_runtime::host::VitaState,
        ) {
            #( #marshal )*
            #ret
        }
    })
}

/// Classify one parameter as state, ctx, or a marshalled value.
fn classify_param(ident: Ident, ty: &Type) -> syn::Result<Param> {
    if let Type::Reference(r) = ty {
        let referent = type_last_ident(&r.elem).map(|i| i.to_string());
        match referent.as_deref() {
            Some("VitaState") => return Ok(Param::State(ident)),
            Some("GuestCtx") => return Ok(Param::Ctx(ident)),
            _ => {
                return Err(syn::Error::new_spanned(
                    ty,
                    "#[hostcall] reference args must be &mut VitaState or &mut GuestCtx",
                ))
            }
        }
    }
    match ValueKind::from_type(ty) {
        Some(kind) => Ok(Param::Value(ident, kind)),
        None => Err(syn::Error::new_spanned(
            ty,
            "#[hostcall] value args must be u32, i32, bool, Ptr, f32, or f64",
        )),
    }
}

/// The expression that reads the next argument of `kind` from the context, using
/// the core register file for integer/pointer args and the VFP file for floats.
fn read_expr(kind: &ValueKind) -> proc_macro2::TokenStream {
    match kind {
        ValueKind::U32 => quote! { __ctx.next_u32() },
        ValueKind::I32 => quote! { __ctx.next_u32() as i32 },
        ValueKind::Bool => quote! { __ctx.next_u32() != 0 },
        ValueKind::Ptr => quote! { ::vitaslop_runtime::Ptr(__ctx.next_u32()) },
        ValueKind::F32 => quote! { __ctx.next_f32() },
        ValueKind::F64 => quote! { __ctx.next_f64() },
    }
}

/// The statement that writes the handler's return value to the right register.
fn write_expr(kind: &ValueKind) -> proc_macro2::TokenStream {
    match kind {
        ValueKind::U32 => quote! { __ctx.ret(__ret); },
        ValueKind::I32 => quote! { __ctx.ret(__ret as u32); },
        ValueKind::Bool => quote! { __ctx.ret(__ret as u32); },
        ValueKind::Ptr => quote! { __ctx.ret(__ret.0); },
        ValueKind::F32 => quote! { __ctx.ret_f32(__ret); },
        ValueKind::F64 => quote! { __ctx.ret_f64(__ret); },
    }
}

/// The final path segment ident of a (possibly qualified) type path, if it is a
/// plain path type. `f32` and `crate::foo::Ptr` both yield their last segment.
fn type_last_ident(ty: &Type) -> Option<&Ident> {
    match ty {
        Type::Path(p) => p.path.segments.last().map(|s| &s.ident),
        _ => None,
    }
}
