#include<bits/stdc++.h>
using namespace std;
char matchpar(char c){
        char res=(c=='}')?'{':((c==']')?'[':'(');
        return res;
}

 bool isValid(string s) {
        
        stack<char>st;
        for(int i=0;i<s.size();i++){
            if(s[i]=='{'||s[i]=='('||s[i]=='[')
            {
                
                st.push(s[i]);
            }
            else if(s[i]==']'||s[i]=='}'||s[i]==')'){
                if(st.top()==matchpar(s[i])){
                    st.pop();
                }
                else{
                    return false;
                }
                
            }
        }
        return true;
        
        
}

int main(){
    string s="[){}";
    if(isValid(s)){
        cout<<"true";
    }
    else{
        cout<<"false";
    }
}