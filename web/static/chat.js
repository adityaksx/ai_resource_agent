async function send(){

    const input = document.getElementById("user-input")
    const msg = input.value

    if(!msg) return

    addMessage("user",msg)

    input.value=""

    const res = await fetch("/chat",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({message:msg})
    })

    const data = await res.json()

    addMessage("bot",data.response)
}


function addMessage(type,text){

    const messages = document.getElementById("messages")

    const div = document.createElement("div")

    div.className = "message " + type

    div.innerText = text

    messages.appendChild(div)

    messages.scrollTop = messages.scrollHeight
}