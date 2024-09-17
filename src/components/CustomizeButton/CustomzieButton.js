function CustomizeButton({ title, onClick }) {
  return (
    <div className="bg-[#1677FF] hover:bg-[#5391e7] w-full shadow py-2 text-center rounded-xl text-white">
      <button onClick={onClick}>{title}</button>
    </div>
  );
}

export default CustomizeButton;
